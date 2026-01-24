import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime
import json

class FitnessDataCollector:
    def __init__(self):
        self.domain = "Fitness and Exercise Science"
        self.data_dir = "collected_fitness_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Better headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def read_local_csv(self, file_path):
        """
        Source 1: Read local CSV file
        """
        print("\n" + "="*70)
        print("SOURCE 1: LOCAL EXERCISE DATABASE (CSV)")
        print("="*70)

        try:
            df = pd.read_csv(file_path, encoding='utf-8')

            print(f"✓ Successfully loaded {len(df)} records")
            print(f"✓ Columns: {len(df.columns)}")
            print(f"\nColumn names: {list(df.columns)}")
            print(f"\nFirst 5 rows:")
            print(df.head())

            # Save to output directory
            output_file = os.path.join(self.data_dir, "exercise_database.csv")
            df.to_csv(output_file, index=False)
            print(f"\n✓ Saved to: {output_file}")

            # Basic statistics
            print(f"\nDataset Statistics:")
            print(f"- Total records: {len(df)}")
            print(f"- Total columns: {len(df.columns)}")

            # Show column-specific stats if certain columns exist
            common_columns = ['Title', 'BodyPart', 'Equipment', 'Type', 'Difficulty', 'Desc']
            for col in common_columns:
                if col in df.columns:
                    print(f"\n{col} distribution:")
                    print(df[col].value_counts().head(10))

            return df

        except FileNotFoundError:
            print(f"✗ Error: File not found at {file_path}")
            print("Please check the file path and try again.")
            return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def scrape_reddit_fitness_json(self):
        """
        Source 2: Scrape Reddit using JSON API
        Reddit provides JSON data by adding .json to URLs
        """
        print("\n" + "="*70)
        print("SOURCE 2: REDDIT r/FITNESS (JSON API)")
        print("="*70)

        # Try multiple approaches
        urls_to_try = [
            "https://www.reddit.com/r/Fitness.json",
            "https://www.reddit.com/r/Fitness/hot.json",
            "https://www.reddit.com/r/Fitness/top.json?t=week"
        ]

        for url in urls_to_try:
            try:
                print(f"\nTrying: {url}")

                # Add Reddit-specific headers
                reddit_headers = self.headers.copy()
                reddit_headers['Accept'] = 'application/json'

                response = requests.get(url, headers=reddit_headers, timeout=15)
                print(f"Response status: {response.status_code}")

                if response.status_code == 200:
                    # Try to parse JSON
                    try:
                        data = response.json()

                        # Extract posts from JSON
                        posts_data = []

                        if 'data' in data and 'children' in data['data']:
                            posts = data['data']['children']
                            print(f"Found {len(posts)} posts in JSON data")

                            for idx, post in enumerate(posts[:20], 1):
                                try:
                                    post_data = post['data']

                                    posts_data.append({
                                        'post_id': idx,
                                        'title': post_data.get('title', ''),
                                        'author': post_data.get('author', ''),
                                        'score': post_data.get('score', 0),
                                        'num_comments': post_data.get('num_comments', 0),
                                        'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                                        'selftext': post_data.get('selftext', '')[:200],  # First 200 chars
                                        'url': f"https://www.reddit.com{post_data.get('permalink', '')}",
                                        'subreddit': post_data.get('subreddit', ''),
                                        'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    })

                                    print(f"  ✓ Post {idx}: {post_data.get('title', '')[:60]}...")

                                except Exception as e:
                                    print(f"  ⚠ Error parsing post {idx}: {e}")
                                    continue

                            if posts_data:
                                df = pd.DataFrame(posts_data)

                                # Save to CSV
                                output_file = os.path.join(self.data_dir, "reddit_posts.csv")
                                df.to_csv(output_file, index=False)

                                print(f"\n✓ Successfully scraped {len(df)} posts")
                                print(f"✓ Saved to: {output_file}")

                                print(f"\nTop 5 posts by score:")
                                print(df.nlargest(5, 'score')[['post_id', 'title', 'score', 'num_comments']])

                                return df

                    except json.JSONDecodeError:
                        print("  ✗ Failed to parse JSON response")
                        continue

                elif response.status_code == 403:
                    print("  ✗ Access forbidden (403)")
                    continue
                elif response.status_code == 429:
                    print("  ✗ Rate limited (429)")
                    time.sleep(2)
                    continue
                else:
                    print(f"  ✗ Unexpected status code: {response.status_code}")
                    continue

            except requests.exceptions.RequestException as e:
                print(f"  ✗ Request error: {e}")
                continue
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        # If all methods fail, try old.reddit.com
        print("\n" + "="*70)
        print("Trying alternative: old.reddit.com")
        print("="*70)

        return self.scrape_old_reddit()

    def scrape_old_reddit(self):
        url = "https://old.reddit.com/r/Fitness/"

        try:
            print(f"Attempting: {url}")

            response = requests.get(url, headers=self.headers, timeout=15)
            print(f"Response status: {response.status_code}")

            if response.status_code != 200:
                print(f"✗ Failed to access old.reddit.com")
                return None

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find posts in old Reddit format
            posts = soup.find_all('div', class_='thing', limit=20)
            print(f"Found {len(posts)} post elements")

            posts_data = []

            for idx, post in enumerate(posts, 1):
                try:
                    # Extract title
                    title_elem = post.find('a', class_='title')
                    if not title_elem:
                        continue

                    title = title_elem.get_text().strip()
                    post_url = title_elem.get('href', '')
                    if post_url.startswith('/'):
                        post_url = f"https://www.reddit.com{post_url}"

                    # Extract score
                    score_elem = post.find('div', class_='score')
                    score = score_elem.get_text().strip() if score_elem else '0'

                    # Extract author
                    author_elem = post.find('a', class_='author')
                    author = author_elem.get_text().strip() if author_elem else ''

                    # Extract comment count
                    comments_elem = post.find('a', class_='comments')
                    comments = comments_elem.get_text().strip() if comments_elem else '0'

                    posts_data.append({
                        'post_id': idx,
                        'title': title,
                        'author': author,
                        'score': score,
                        'comments': comments,
                        'url': post_url,
                        'subreddit': 'r/Fitness',
                        'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

                    print(f"  ✓ Post {idx}: {title[:60]}...")

                except Exception as e:
                    continue

            if posts_data:
                df = pd.DataFrame(posts_data)

                output_file = os.path.join(self.data_dir, "reddit_posts.csv")
                df.to_csv(output_file, index=False)

                print(f"\n✓ Successfully scraped {len(df)} posts")
                print(f"✓ Saved to: {output_file}")

                return df
            else:
                print("✗ No posts extracted")
                return None

        except Exception as e:
            print(f"✗ Error: {e}")
            return None

    def fetch_pubmed_articles(self):
        """
        Source 3: Fetch articles from PubMed API
        """
        print("\n" + "="*70)
        print("SOURCE 3: PUBMED RESEARCH ARTICLES (API)")
        print("="*70)

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_terms = ["resistance training", "exercise physiology", "strength training"]

        all_articles = []

        for term in search_terms:
            try:
                print(f"\nSearching PubMed for: '{term}'")

                # Search for articles
                search_url = f"{base_url}esearch.fcgi"
                search_params = {
                    'db': 'pubmed',
                    'term': term,
                    'retmax': 5,
                    'retmode': 'json',
                    'sort': 'relevance'
                }

                search_response = requests.get(search_url, params=search_params, timeout=15)

                if search_response.status_code != 200:
                    print(f"  ✗ Search failed (Status: {search_response.status_code})")
                    continue

                search_data = search_response.json()
                id_list = search_data.get('esearchresult', {}).get('idlist', [])

                print(f"  Found {len(id_list)} article IDs")

                if not id_list:
                    continue

                # Fetch article summaries
                summary_url = f"{base_url}esummary.fcgi"
                summary_params = {
                    'db': 'pubmed',
                    'id': ','.join(id_list),
                    'retmode': 'json'
                }

                summary_response = requests.get(summary_url, params=summary_params, timeout=15)

                if summary_response.status_code != 200:
                    print(f"  ✗ Summary fetch failed")
                    continue

                summary_data = summary_response.json()

                # Extract article information
                for pmid in id_list:
                    try:
                        article = summary_data['result'][pmid]

                        # Extract authors
                        authors_list = article.get('authors', [])
                        authors = ', '.join([a.get('name', '') for a in authors_list[:3]])
                        if len(authors_list) > 3:
                            authors += ' et al.'

                        article_data = {
                            'pmid': pmid,
                            'title': article.get('title', ''),
                            'authors': authors,
                            'journal': article.get('fulljournalname', ''),
                            'pub_date': article.get('pubdate', ''),
                            'source': article.get('source', ''),
                            'search_term': term,
                            'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                        all_articles.append(article_data)
                        print(f"  ✓ PMID {pmid}: {article.get('title', '')[:60]}...")

                    except Exception as e:
                        print(f"  ✗ Error parsing article {pmid}: {e}")
                        continue

                time.sleep(0.5)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        if len(all_articles) == 0:
            print("\n✗ No articles fetched from PubMed")
            return None

        df = pd.DataFrame(all_articles)

        output_file = os.path.join(self.data_dir, "pubmed_articles.csv")
        df.to_csv(output_file, index=False)

        print(f"\n✓ Successfully fetched {len(df)} articles")
        print(f"✓ Saved to: {output_file}")

        print(f"\nArticles by search term:")
        print(df['search_term'].value_counts())

        return df

    def generate_summary(self):
        """
        Generate summary of collected data
        """
        print("\n" + "="*70)
        print("COLLECTION SUMMARY")
        print("="*70)

        summary_data = []

        csv_files = [
            'exercise_database.csv',
            'reddit_posts.csv',
            'pubmed_articles.csv'
        ]

        total_records = 0
        successful_sources = 0

        for filename in csv_files:
            filepath = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    file_size = os.path.getsize(filepath) / 1024

                    summary_data.append({
                        'source': filename.replace('.csv', '').replace('_', ' ').title(),
                        'filename': filename,
                        'records': len(df),
                        'columns': len(df.columns),
                        'size_kb': round(file_size, 2),
                        'status': 'Success ✓'
                    })

                    total_records += len(df)
                    successful_sources += 1

                except Exception as e:
                    summary_data.append({
                        'source': filename.replace('.csv', '').replace('_', ' ').title(),
                        'filename': filename,
                        'records': 0,
                        'columns': 0,
                        'size_kb': 0,
                        'status': f'Error ✗'
                    })
            else:
                summary_data.append({
                    'source': filename.replace('.csv', '').replace('_', ' ').title(),
                    'filename': filename,
                    'records': 0,
                    'columns': 0,
                    'size_kb': 0,
                    'status': 'Not collected ✗'
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)

            summary_file = os.path.join(self.data_dir, "collection_summary.csv")
            summary_df.to_csv(summary_file, index=False)

            print(f"\nCollection Results:")
            print(f"- Total sources attempted: 3")
            print(f"- Successful collections: {successful_sources}/3")
            print(f"- Total records collected: {total_records}")

            print(f"\nDetailed breakdown:")
            print(summary_df.to_string(index=False))

            print(f"\n✓ Summary saved to: {summary_file}")

            return summary_df

        return None


def main():
    """
    Main execution
    """
    print("="*70)
    print("DSCI-560 LAB 2 - FITNESS DATA COLLECTION")

    print("="*70)

    collector = FitnessDataCollector()

    # Source 1: Local CSV
    print("\n" + "="*70)
    csv_path = r"C:\Users\闫世达\Desktop\USC课程\DSCI560\Lab\data\megaGymDataset.csv"
    print("="*70)

    exercise_df = collector.read_local_csv(csv_path)

    # Source 2: Reddit
    reddit_df = collector.scrape_reddit_fitness_json()

    # Source 3: PubMed
    pubmed_df = collector.fetch_pubmed_articles()

    # Generate summary
    summary_df = collector.generate_summary()

    print("\n" + "="*70)
    print("COLLECTION COMPLETED")
    print("="*70)

    print(f"\nAll data saved in: {collector.data_dir}/")

    if summary_df is not None:
        successful = summary_df[summary_df['status'].str.contains('Success')]['source'].tolist()
        if successful:
            print(f"\nSuccessfully collected from:")
            for source in successful:
                print(f"  ✓ {source}")


if __name__ == "__main__":
    main()