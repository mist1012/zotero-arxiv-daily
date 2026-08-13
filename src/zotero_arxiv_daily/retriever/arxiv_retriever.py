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
# Only papers associated with the following conferences/journals
# are allowed to continue to full-text processing and reranking.
# ============================================================

VENUE_PATTERNS = {
    # --------------------------------------------------------
    # Robotics / Embodied AI conferences
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Computer Vision conferences
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Robotics journals
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Computer Vision journals
    # --------------------------------------------------------
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


# ============================================================
# Papers of these types do NOT count.
#
# Even if "CVPR", "ICRA", etc. appears in the comment,
# workshop/demo/extended abstract papers are rejected.
# ============================================================

EXCLUDED_PAPER_TYPES = re.compile(
    r"\bworkshop\b"
    r"|\bdemo\b"
    r"|\bdemonstration\b"
    r"|\bextended abstract\b",
    re.IGNORECASE,
)


# ============================================================
# Explicit evidence that the paper is accepted / published.
# ============================================================

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


# ============================================================
# Explicit evidence that the paper has NOT yet been accepted.
# ============================================================

REJECTED_MARKERS = re.compile(
    r"\bsubmitted\b"
    r"|\bsubmission\b"
    r"|\bunder review\b"
    r"|\bto be submitted\b"
    r"|\bmanuscript submitted\b",
    re.IGNORECASE,
)


# ============================================================
# Conferences for which:
#
#     "IROS 2026"
#     "ICRA 2026"
#     "CVPR 2026"
#
# is considered sufficient evidence,
# as long as submitted / under review / workshop etc.
# does NOT appear.
#
# Journals remain stricter.
# ============================================================

CONFERENCE_VENUES = {
    "ICRA",
    "IROS",
    "RSS",
    "CoRL",
    "CVPR",
    "ICCV",
    "ECCV",
}


def _match_whitelist_venue(text: str) -> str | None:
    """
    Match text against the venue whitelist.

    Examples:

        "Accepted to IROS 2026"
            -> "IROS"

        "IEEE Robotics and Automation Letters"
            -> "IEEE RA-L"

        "European Conference on Computer Vision 2026"
            -> "ECCV"

    Returns None if no whitelisted venue is found.
    """
    if not text:
        return None

    for venue, pattern in VENUE_PATTERNS.items():
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return venue

    return None


def _extract_year(text: str) -> str | None:
    """
    Extract a four-digit publication year.

    Example:
        "IROS 2026" -> "2026"
    """
    if not text:
        return None

    match = re.search(
        r"\b(20\d{2})\b",
        text,
    )

    if match:
        return match.group(1)

    return None


def _canonical_venue_with_year(
    venue: str,
    text: str,
) -> str:
    """
    Add year to conference venue names when available.

    Examples:

        IROS + "IROS 2026"
            -> "IROS 2026"

        ECCV + "Accepted to ECCV 2026"
            -> "ECCV 2026"

        IEEE RA-L
            -> "IEEE RA-L"
    """
    if venue not in CONFERENCE_VENUES:
        return venue

    year = _extract_year(text)

    if year:
        return f"{venue} {year}"

    return venue


def _confirmed_venue_from_comment(
    comment: str,
) -> str | None:
    """
    Inspect the arXiv comment.

    Conference rules
    ----------------

    PASS:

        Accepted to IROS 2026

        Accepted at ICRA 2026

        ECCV 2026 Oral

        CVPR 2026 Highlight

        IEEE/RSJ International Conference on
        Intelligent Robots and Systems (IROS 2026)

        IROS 2026


    REJECT:

        Submitted to IROS 2026

        Under review at CVPR 2026

        CVPR 2026 Workshop

        ICRA Workshop 2026

        IROS 2026 Demo Paper


    Important:

    For conferences, an explicit venue + year is enough
    even if the word "accepted" is absent.

    For journals, merely mentioning the journal name in
    the comment is NOT enough. Explicit accepted/published
    evidence is still required.

    Formal journal_ref metadata is handled separately in
    get_confirmed_venue().
    """
    if not comment:
        return None

    # --------------------------------------------------------
    # Reject workshop/demo/extended abstract globally.
    # --------------------------------------------------------

    if EXCLUDED_PAPER_TYPES.search(comment):
        return None

    # --------------------------------------------------------
    # Comments can contain several statements.
    #
    # We inspect them separately so something like:
    #
    #   "Accepted elsewhere. Submitted to IROS 2026."
    #
    # cannot accidentally become IROS PASS.
    # --------------------------------------------------------

    segments = re.split(
        r"[.;\n]+",
        comment,
    )

    for segment in segments:
        segment = segment.strip()

        if not segment:
            continue

        venue = _match_whitelist_venue(
            segment
        )

        if venue is None:
            continue

        # ----------------------------------------------------
        # Submitted / Under review always loses.
        # ----------------------------------------------------

        if REJECTED_MARKERS.search(segment):
            continue

        # ----------------------------------------------------
        # Explicit acceptance/publication always passes.
        # ----------------------------------------------------

        if ACCEPTED_MARKERS.search(segment):
            return _canonical_venue_with_year(
                venue,
                segment,
            )

        # ----------------------------------------------------
        # NEW RULE:
        #
        # Conference + explicit year also passes.
        #
        # Examples:
        #
        #   IROS 2026
        #
        #   IEEE/RSJ International Conference on Intelligent
        #   Robots and Systems (IROS 2026)
        #
        #   CVPR 2026
        #
        # But:
        #
        #   Submitted to IROS 2026
        #
        # was already rejected above.
        # ----------------------------------------------------

        if venue in CONFERENCE_VENUES:
            year = _extract_year(
                segment
            )

            if year is not None:
                return f"{venue} {year}"

    return None


def get_confirmed_venue(
    paper: ArxivResult,
) -> str | None:
    """
    Strict venue filtering.

    Rules:

    1. journal_ref is considered formal publication evidence
       if it matches the whitelist.

    2. For arXiv comments:
         - accepted / published / to appear -> PASS
         - conference + year -> PASS
         - submitted / under review -> REJECT
         - workshop / demo -> REJECT

    3. Journals in comments remain strict:
       explicit acceptance/publication wording is required.

    Returns:
        Canonical venue string or None.
    """

    journal_ref = (
        getattr(
            paper,
            "journal_ref",
            None,
        )
        or ""
    )

    comment = (
        getattr(
            paper,
            "comment",
            None,
        )
        or ""
    )

    # --------------------------------------------------------
    # First check formal journal_ref.
    #
    # journal_ref normally indicates that publication metadata
    # has already been registered on arXiv.
    # --------------------------------------------------------

    if journal_ref:
        if not EXCLUDED_PAPER_TYPES.search(
            journal_ref
        ):
            venue = _match_whitelist_venue(
                journal_ref
            )

            if venue is not None:
                return _canonical_venue_with_year(
                    venue,
                    journal_ref,
                )

    # --------------------------------------------------------
    # Otherwise inspect arXiv comments.
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

        with open(
            path,
            "wb",
        ) as file:
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
        # Get latest papers from arXiv RSS feed.
        # ----------------------------------------------------

        feed = feedparser.parse(
            f"https://rss.arxiv.org/atom/{query}"
        )

        if (
            "Feed error for query"
            in feed.feed.title
        ):
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
            entry.id.removeprefix(
                "oai:arXiv.org:"
            )
            for entry in feed.entries
            if entry.get(
                "arxiv_announce_type",
                "new",
            )
            in allowed_announce_types
        ]

        # ----------------------------------------------------
        # Test/debug only examines the first 10 papers.
        # ----------------------------------------------------

        if self.config.executor.debug:
            all_paper_ids = (
                all_paper_ids[:10]
            )

        # ----------------------------------------------------
        # Retrieve complete arXiv metadata.
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
        # Filtering happens BEFORE convert_to_paper().
        #
        # Therefore rejected papers will NOT:
        #
        # - download TeX
        # - download HTML
        # - download PDF
        # - extract full text
        # - enter Zotero reranking
        #
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
        # Final fallback: PDF.
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
