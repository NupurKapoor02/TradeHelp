import requests
from bs4 import BeautifulSoup

def get_nse_stock_symbols():
    # Define the NSE URL containing the list of stocks
    nse_url = 'https://www1.nseindia.com/products/content/equities/equities/eq_security.htm'

    # Set headers to mimic a web browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    }

    # Send a GET request to the NSE website
    response = requests.get(nse_url, headers=headers)

    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table containing stock symbols
        table = soup.find('table', {'id': 'tablesorter1'})

        # Extract stock symbols from the table
        symbols = [row.td.text.strip() for row in table.find_all('tr') if row.td]

        # Remove the table header
        symbols = symbols[1:]

        return symbols
    else:
        print(f"Failed to retrieve data. Status Code: {response.status_code}")
        return None


stock_symbols = get_nse_stock_symbols()
if stock_symbols:
    for symbol in stock_symbols:
        print(symbol)