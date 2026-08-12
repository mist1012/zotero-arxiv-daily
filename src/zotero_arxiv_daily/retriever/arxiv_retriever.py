from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
import re
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests


T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


# ============================================================
# Strict venue filter
#
# Only papers that can be clearly confirmed as accepted by or
# published in the following conferences/journals will pass.
# ============================================================

VENUE_PATTERNS = {
    # Robotics / Embodied AI conferences
    "ICRA": (
        r"\bICRA\b"
        r"|International Conference on Robotics and Automation"
    ),
    "IROS": (
        r"\bIROS\b"
        r"|International Conference on Intelligent Robots and Systems"
    ),
    "RSS": (
        r"\bRSS\b"
        r"|Robotics\s*:\s*Science and Systems"
        r"|Robotics Science and Systems"
    ),
    "CoRL": (
        r"\bCoRL\b"
        r"|Conference on Robot Learning"
    ),

    # Computer Vision conferences
    "CVPR": (
        r"\bCVPR\b"
        r"|Conference on Computer Vision and Pattern Recognition"
    ),
    "ICCV": (
        r"\bICCV\b"
        r"|International Conference on Computer Vision"
    ),
    "ECCV": (
        r"\bECCV\b"
        r"|European Conference on Computer Vision"
    ),

    # Robotics journals
    "IEEE RA-L": (
        r"\bRA-?L\b"
        r"|\bRAL\b"
        r"|Robotics and Automation Letters"
    ),
    "IEEE T-RO": (
        r"\bT-?RO\b"
        r"|\bTRO\b"
        r"|Transactions on Robotics"
    ),
    "IJRR": (
        r"\bIJRR\b"
        r"|International Journal of Robotics Research"
    ),
    "Science Robotics": (
        r"Science Robotics"
    ),

    # Computer Vision journals
    "TPAMI": (
        r"\bTPAMI\b"
        r"|\bPAMI\b"
        r"|Transactions on Pattern Analysis and Machine Intelligence"
    ),
    "IJCV": (
        r"\bIJCV\b"
        r"|International Journal of Computer Vision"
    ),
}


# Workshop / Demo / Extended Abstract do NOT count as
# main-conference or journal papers.
EXCLUDED_PAPER_TYPES = re.compile(
    r"\bworkshop\b"
    r"|\bdemo\b"
    r"|\bdemonstration\b"
    r"|\bextended abstract\b",
    re.IGNORECASE,
)


# Explicit evidence that the paper has already been accepted
# or published.
ACCEPTED_MARKERS = re.compile(
    r"\baccepted\b"
    r"|\bto appear\b"
    r"|\bwill appear\b"
    r"|\bpublished\b"
    r"|\boral\b"
    r"|\bspotlight\b"
    r"|\bhighlight\b",
    re.IGNORECASE,
)


# Evidence that the paper is only submitted / under review.
REJECTED_MARKERS = re.compile(
    r"\bsubmitted\b"
    r"|\bsubmission\b"
    r"|\bunder review\b"
    r"|\bto be submitted\b"
    r"|\bmanuscript submitted\b",
    re.IGNORECASE,
)


def _match_whitelist_venue(text: str) -> str | None:
    """
    Match venue metadata against the whitelist.

    Returns canonical venue name, e.g.:
        IROS
        ECCV
        IEEE RA-L

    Returns None if no whitelisted venue is found.
    """
    if not text:
        return None

    for venue, pattern in VENUE_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return venue

    return None


def _extract_year(text: str) -> str | None:
    """
    Try to extract a publication year from venue metadata.
    """
    if not text:
        return None

    match = re.search(r"\b(20\d{2})\b", text)

    if match:
        return match.group(1)

    return None


def _canonical_venue_with_year(
    venue: str,
    text: str,
) -> str:
    """
    Add conference year when available.

    Examples:
        IROS + "... IROS 2026 ..." -> "IROS 2026"
        ECCV + "... ECCV 2026 ..." -> "ECCV 2026"

    Journal names are kept without appending a year.
    """
    conference_venues = {
        "ICRA",
        "IROS",
        "RSS",
        "CoRL",
        "CVPR",
        "ICCV",
        "ECCV",
    }

    if venue not in conference_venues:
        return venue

    year = _extract_year(text)

    if year:
        return f"{venue} {year}"

    return venue


def _confirmed_venue_from_comment(
    comment: str,
) -> str | None:
    """
    Conservatively inspect the arXiv comment.

    Requirements:

    1. Venue must be in the whitelist.
    2. The same segment must contain evidence such as
       "accepted", "to appear", "published", etc.
    3. "submitted", "under review", etc. do NOT count.
    4. Workshop / Demo / Extended Abstract are rejected.

    This prevents cases such as:

        "Accepted elsewhere. Submitted to IROS 2026"

    from being incorrectly classified as an IROS paper.
    """
    if not comment:
        return None

    # Reject workshop/demo-style papers globally.
    if EXCLUDED_PAPER_TYPES.search(comment):
        return None

    # Split comments into smaller segments so acceptance evidence
    # must appear close to the matched venue.
    segments = re.split(r"[.;\n]+", comment)

    for segment in segments:
        venue = _match_whitelist_venue(segment)

        if venue is None:
            continue

        # Explicit submission/under-review language means this
        # segment is not proof of acceptance.
        if REJECTED_MARKERS.search(segment):
            continue

        # Require explicit acceptance/publication evidence.
        if ACCEPTED_MARKERS.search(segment):
            return _canonical_venue_with_year(
                venue,
                segment,
            )

    return None


def get_confirmed_venue(
    paper: ArxivResult,
) -> str | None:
    """
    Strict venue filtering.

    Rules:

    1. journal_ref is regarded as formal publication evidence
       if it matches the whitelist.

    2. arXiv comment must explicitly state accepted /
       published / to appear / oral / spotlight / highlight.

    3. Workshop / Demo / Extended Abstract are rejected.

    4. Submitted / Under Review papers are rejected.

    Returns:
        canonical venue name if confirmed, otherwise None.
    """
    journal_ref = (
        getattr(paper, "journal_ref", None)
        or ""
    )

    comment = (
        getattr(paper, "comment", None)
        or ""
    )

    # --------------------------------------------------------
    # First check journal_ref.
    #
    # journal_ref normally indicates formal publication, so
    # explicit "accepted" wording is not required here.
    # --------------------------------------------------------

    if journal_ref:
        if not EXCLUDED_PAPER_TYPES.search(journal_ref):
            venue = _match_whitelist_venue(
                journal_ref
            )

            if venue is not None:
                return _canonical_venue_with_year(
                    venue,
                    journal_ref,
                )

    # --------------------------------------------------------
    # Otherwise inspect arXiv comment using strict rules.
    # --------------------------------------------------------

    return _confirmed_venue_from_comment(
        comment
    )


def _download_file(
    url: str,
    path: str,
) -> None:
    with requests.get(
        url,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    ) as response:
        response.raise_for_status()

        with open(path, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(
            (
                "ok",
                func(*args),
            )
        )

    except Exception as exc:
        result_queue.put(
            (
                "error",
                f"{type(exc).__name__}: {exc}",
            )
        )


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = (
        multiprocessing.get_all_start_methods()
    )

    context = multiprocessing.get_context(
        "fork"
        if "fork" in start_methods
        else start_methods[0]
    )

    result_queue = context.Queue()

    process = context.Process(
        target=_run_in_subprocess,
        args=(
            result_queue,
            func,
            args,
        ),
    )

    process.start()

    try:
        status, payload = result_queue.get(
            timeout=timeout
        )

    except Empty:
        if process.is_alive():
            process.kill()

        process.join(5)

        result_queue.close()
        result_queue.join_thread()

        logger.warning(
            f"{operation} timed out for "
            f"{paper_title} after "
            f"{timeout} seconds"
        )

        return None

    process.join(5)

    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(
        f"{operation} failed for "
        f"{paper_title}: {payload}"
    )

    return None


def _extract_text_from_pdf_worker(
    pdf_url: str,
) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(
            temp_dir,
            "paper.pdf",
        )

        _download_file(
            pdf_url,
            path,
        )

        return extract_markdown_from_pdf(
            path
        )


def _extract_text_from_html_worker(
    html_url: str,
) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(
        html_url
    )

    if downloaded is None:
        raise ValueError(
            f"Failed to download HTML from "
            f"{html_url}"
        )

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
    )

    if not text:
        raise ValueError(
            f"No text extracted from "
            f"{html_url}"
        )

    return text


def _extract_text_from_tar_worker(
    source_url: str,
    paper_id: str,
    paper_title: str | None = None,
) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(
            temp_dir,
            "paper.tar.gz",
        )

        _download_file(
            source_url,
            path,
        )

        file_contents = extract_tex_code_from_tar(
            path,
            paper_id,
            paper_title=paper_title,
        )

        if (
            not file_contents
            or "all" not in file_contents
        ):
            raise ValueError(
                "Main tex file not found."
            )

        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):

    def __init__(
        self,
        config,
    ):
        super().__init__(config)

        if self.config.source.arxiv.category is None:
            raise ValueError(
                "category must be specified for arxiv."
            )

    def _retrieve_raw_papers(
        self,
    ) -> list[ArxivResult]:
        client = arxiv.Client(
            num_retries=10,
            delay_seconds=10,
        )

        query = "+".join(
            self.config.source.arxiv.category
        )

        include_cross_list = (
            self.config.source.arxiv.get(
                "include_cross_list",
                False,
            )
        )

        # ----------------------------------------------------
        # Get latest papers from arXiv RSS feed
        # ----------------------------------------------------

        feed = feedparser.parse(
            f"https://rss.arxiv.org/atom/{query}"
        )

        if "Feed error for query" in feed.feed.title:
            raise Exception(
                f"Invalid ARXIV_QUERY: {query}."
            )

        raw_papers = []

        allowed_announce_types = (
            {"new", "cross"}
            if include_cross_list
            else {"new"}
        )

        all_paper_ids = [
            i.id.removeprefix(
                "oai:arXiv.org:"
            )
            for i in feed.entries
            if i.get(
                "arxiv_announce_type",
                "new",
            )
            in allowed_announce_types
        ]

        # ----------------------------------------------------
        # Debug/Test mode only retrieves first 10 papers.
        # ----------------------------------------------------

        if self.config.executor.debug:
            all_paper_ids = (
                all_paper_ids[:10]
            )

        # ----------------------------------------------------
        # Get complete metadata from arXiv API
        # ----------------------------------------------------

        bar = tqdm(
            total=len(all_paper_ids)
        )

        max_batch_retries = 5
        batch_retry_delay = 30

        for i in range(
            0,
            len(all_paper_ids),
            20,
        ):
            search = arxiv.Search(
                id_list=all_paper_ids[
                    i:i + 20
                ]
            )

            for attempt in range(
                max_batch_retries
            ):
                try:
                    batch = list(
                        client.results(search)
                    )

                    bar.update(
                        len(batch)
                    )

                    raw_papers.extend(
                        batch
                    )

                    break

                except arxiv.HTTPError as exc:
                    if (
                        exc.status == 429
                        and attempt
                        < max_batch_retries - 1
                    ):
                        wait = (
                            batch_retry_delay
                            * (attempt + 1)
                        )

                        logger.warning(
                            "arXiv API 429 on "
                            f"batch {i // 20}, "
                            f"retry "
                            f"{attempt + 1}/"
                            f"{max_batch_retries} "
                            f"in {wait}s"
                        )

                        sleep(wait)

                    else:
                        raise

            if i + 20 < len(all_paper_ids):
                sleep(3)

        bar.close()

        # ====================================================
        # STRICT VENUE FILTER
        #
        # IMPORTANT:
        #
        # Filtering happens BEFORE convert_to_paper().
        #
        # Rejected papers therefore do NOT:
        # - download TeX
        # - download HTML
        # - download PDF
        # - extract full text
        # - enter reranking
        #
        # This greatly reduces GitHub Actions runtime.
        # ====================================================

        before_count = len(
            raw_papers
        )

        qualified_papers = []

        logger.info(
            "Venue strict filter enabled."
        )

        logger.info(
            "Allowed venues: "
            + ", ".join(
                VENUE_PATTERNS.keys()
            )
        )

        for paper in raw_papers:
            venue = get_confirmed_venue(
                paper
            )

            if venue is not None:
                qualified_papers.append(
                    paper
                )

                logger.info(
                    f"Venue PASS "
                    f"[{venue}] - "
                    f"{paper.title}"
                )

            elif self.config.executor.debug:
                comment = (
                    getattr(
                        paper,
                        "comment",
                        None,
                    )
                    or ""
                )

                journal_ref = (
                    getattr(
                        paper,
                        "journal_ref",
                        None,
                    )
                    or ""
                )

                logger.info(
                    "Venue REJECT - "
                    f"{paper.title} | "
                    f"comment={comment!r} | "
                    f"journal_ref="
                    f"{journal_ref!r}"
                )

        logger.info(
            "Venue strict filter: "
            f"{before_count} raw arXiv "
            f"papers -> "
            f"{len(qualified_papers)} "
            f"qualified papers"
        )

        if len(qualified_papers) == 0:
            logger.info(
                "No papers matched the "
                "strict venue whitelist."
            )

        return qualified_papers

    def convert_to_paper(
        self,
        raw_paper: ArxivResult,
    ) -> Paper:
        title = raw_paper.title

        authors = [
            author.name
            for author
            in raw_paper.authors
        ]

        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url

        # ----------------------------------------------------
        # Try TeX source first.
        # ----------------------------------------------------

        full_text = extract_text_from_tar(
            raw_paper
        )

        # ----------------------------------------------------
        # Fall back to arXiv HTML.
        # ----------------------------------------------------

        if full_text is None:
            full_text = (
                extract_text_from_html(
                    raw_paper
                )
            )

        # ----------------------------------------------------
        # Final fallback: download PDF.
        # ----------------------------------------------------

        if full_text is None:
            full_text = (
                extract_text_from_pdf(
                    raw_paper
                )
            )

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(
    paper: ArxivResult,
) -> str | None:
    html_url = paper.entry_id.replace(
        "/abs/",
        "/html/",
    )

    try:
        return (
            _extract_text_from_html_worker(
                html_url
            )
        )

    except Exception as exc:
        logger.warning(
            "HTML extraction failed for "
            f"{paper.title}: {exc}"
        )

        return None


def extract_text_from_pdf(
    paper: ArxivResult,
) -> str | None:
    if paper.pdf_url is None:
        logger.warning(
            "No PDF URL available for "
            f"{paper.title}"
        )

        return None

    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (
            paper.pdf_url,
        ),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(
    paper: ArxivResult,
) -> str | None:
    source_url = paper.source_url()

    if source_url is None:
        logger.warning(
            "No source URL available for "
            f"{paper.title}"
        )

        return None

    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (
            source_url,
            paper.entry_id,
            paper.title,
        ),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
