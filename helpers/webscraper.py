import requests
from bs4 import BeautifulSoup
import os, urllib

base_url = ''
start_url = f'{base_url}/latest/index.html'
download_folder = ''

os.makedirs(download_folder, exist_ok=True)

def get_soup(url):
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, 'html.parser')

def download_page(url):
    soup = get_soup(url)
    page_path = url.replace(base_url, '').lstrip('/')
    page_folder, page_filename = os.path.split(page_path)
    os.makedirs(os.path.join(download_folder, page_folder), exist_ok=True)
    
    with open(os.path.join(download_folder, page_path), 'w', encoding='utf-8') as f:
        f.write(str(soup))
    
    return soup
def find_child_pages(soup):
    all_links = [link.get('href') for link in soup.find_all('a') if link.get('href')]
    child_links = [urllib.parse.urljoin(base_url, link) for link in all_links if not link.startswith('#')]
    return child_links

if __name__ == '__main__':
    # Download the start page
    soup = download_page(start_url)

    # Find and download child pages
    child_pages = find_child_pages(soup)
    for child_page in child_pages:
        print(f"Downloading: {child_page}")  # To help debug
        try:
            download_page(child_page)
        except Exception as e:
            print(f"Failed to download {child_page}: {e}")

    print("Download completed")
