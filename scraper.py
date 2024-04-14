import requests
from bs4 import BeautifulSoup

def get_website_text(url):
  """Fetches the text content from a website using requests and Beautiful Soup.

  Args:
      url: The URL of the website.

  Returns:
      The extracted text content of the website.
  """
  try:
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for unsuccessful requests
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(separator='\n')  # Combine text with newlines
    return text
  except requests.exceptions.RequestException as e:
    print(f"Error fetching website: {e}")
    return None

# Example usage
website_url = "https://notion.com"  # Replace with the target website
text_content = get_website_text(website_url)

if text_content:
  print("Extracted Text:")
  print(text_content)
else:
  print("Failed to retrieve text content.")
