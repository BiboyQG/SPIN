import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from typing import Dict, Optional, List
from urllib.parse import urljoin
import re


class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        # Common class and ID names for navigation elements
        self.nav_selectors = [
            "nav",
            "navigation",
            "navbar",
            "header",
            "menu",
            "topbar",
            ".nav",
            ".navigation",
            ".navbar",
            ".header",
            ".menu",
            "#nav",
            "#navigation",
            "#navbar",
            "#header",
            "#menu",
            "[role='navigation']",
        ]

        # Common class and ID names for footer elements
        self.footer_selectors = [
            "footer",
            ".footer",
            "#footer",
            ".bottom",
            "#bottom",
            ".site-footer",
            "#site-footer",
            "[role='contentinfo']",
        ]

        # Add these to the __init__ method if needed
        self.nav_selectors.extend(
            [
                ".menucol",  # The navigation menu columns
                "#main-nav",
                ".dropdown",  # Dropdown menus
                ".main-header",
            ]
        )

        self.footer_selectors.extend(
            [
                ".site-footer",
                ".footer-content",
                "#footer-wrapper",
            ]
        )

    def remove_elements(self, soup: BeautifulSoup, selectors: List[str]) -> None:
        """Remove elements matching any of the given selectors."""
        for selector in selectors:
            for element in soup.select(selector):
                element.decompose()

    def clean_markdown(self, content: str) -> str:
        """Clean up markdown content by adding appropriate spacing and removing excessive blank lines."""
        # First, normalize all newlines and remove extra spaces
        content = re.sub(r"\s*\n\s*", "\n", content)

        # Ensure proper spacing around headers
        content = re.sub(r"(^|\n)(#+ .*?)(\n|$)", r"\n\n\2\n\n", content)

        # Add spacing around bullet points
        content = re.sub(r"(^|\n)(\* .*?)(\n|$)", r"\n\n\2\n\n", content)

        # Add spacing around standalone links
        content = re.sub(r"(^|\n)(\[.*?\]\(.*?\))(\n|$)", r"\n\n\2\n\n", content)

        # Handle contact information (phone, email) with single line breaks
        content = re.sub(r"(\(\d{3}\) \d{3}-\d{4})\n\n", r"\1\n", content)
        content = re.sub(r"(\[.*?@.*?\]\(mailto:.*?\))\n\n", r"\1\n", content)

        # Clean up multiple blank lines (more than 2) into double blank lines
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Remove blank lines at start and end
        content = re.sub(r"^\n+", "", content)
        content = re.sub(r"\n+$", "", content)

        # Special handling for title/header at the start
        if content.startswith("# "):
            content = "# " + content[2:].lstrip()

        return content

    def scrape_url(self, url: str, params: Optional[Dict] = None) -> Dict[str, str]:
        """
        Scrape a URL and return its content in markdown format.

        Args:
            url (str): The URL to scrape
            params (Optional[Dict]): Additional parameters (kept for compatibility with firecrawl)

        Returns:
            Dict[str, str]: Dictionary containing the markdown content
        """
        try:
            # Fetch the webpage
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove navigation elements
            self.remove_elements(soup, self.nav_selectors)

            # Remove footer elements
            self.remove_elements(soup, self.footer_selectors)

            # Remove unwanted tags
            for tag in soup.find_all(["img", "video", "script", "style"]):
                tag.decompose()

            # Fix relative URLs to absolute URLs
            for a in soup.find_all("a", href=True):
                a["href"] = urljoin(url, a["href"])

            # Convert to markdown using markdownify
            markdown_content = md(str(soup), heading_style="ATX")

            # Clean up the markdown content
            markdown_content = self.clean_markdown(markdown_content)

            return {"markdown": markdown_content}

        except requests.RequestException as e:
            raise Exception(f"Error scraping URL {url}: {str(e)}")
