from paperswithcode import PapersWithCodeClient

class PapersRepositoryFetcher:
    def __init__(self, client: PapersWithCodeClient, items_per_page: int = 50):
        self.client = client
        self.items_per_page = items_per_page

    def get_official_paper_repos(self):
        page = 1
        while True:
            paper_repos = self.client.search(page=page, items_per_page=self.items_per_page)
            for paper_repo in paper_repos.results:
                if paper_repo.is_official:
                    paper = paper_repo.paper
                    repo = paper_repo.repository
                    if paper.url_pdf and repo.url:
                        yield paper.url_pdf, repo.url
            if not paper_repos.next_page:
                break
            page = paper_repos.next_page

    @staticmethod
    def transform_to_https_git_link(url):
        return url.rstrip('/') + ('.git' if not url.endswith('.git') else '')

    def run(self):
        official_paper_repos = list(self.get_official_paper_repos())
        
        # Store the PDF and repository URLs with .git suffix
        with open("downloader/official_paper_repos.txt", "w") as f:
            for pdf_url, repo_url in official_paper_repos:
                # Ensure the repository URL ends with .git
                modified_repo_url = self.transform_to_https_git_link(repo_url)
                f.write(f"{pdf_url}\t{modified_repo_url}\n")
        
        print(f"Found {len(official_paper_repos)} official paper-repository pairs.")

if __name__ == '__main__':
    client = PapersWithCodeClient()
    fetcher = PapersRepositoryFetcher(client)
    fetcher.run()
