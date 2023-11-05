import requests
from bs4 import BeautifulSoup

# Define the URL of the RapidAPI Sports category page
url = "https://rapidapi.com/category/Sports"

# Send an HTTP GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # You can now work with the parsed HTML content using BeautifulSoup
    # For example, let's extract and print the title of the page:
    title = soup.title.string
    print(f"Page Title: {title}")

        # You can also extract other information from the page as needed

    # Find all the div elements with class 'ItemCard'
    item_cards = soup.find_all('div', class_='ItemCard')

    # Initialize empty lists to store the extracted information
    api_names = []
    api_requests = []
    performance_values = []
    response_times = []
    completion_rates = []

    # Iterate through the 'ItemCard' divs
    for item_card in item_cards:
        # Find the 'ApiName' div inside the 'ItemCard'
        api_name = item_card.find('div', class_='ApiName')
        api_names.append(api_name.text.strip() if api_name else 'N/A')

        # Find the 'a' tag inside the 'ItemCard' to get the request entry
        api_link = item_card.find('a', class_='CardLink')
        api_request_entry = api_link['href'] if api_link else 'N/A'
        api_requests.append(api_request_entry)

        # Find all the div elements with class 'Metrics' inside the 'ItemCard'
        metrics_divs = item_card.find_all('div', class_='Metrics')

        # Extract and append the data based on the order in the HTML
        for metrics_div in metrics_divs:
            captions = metrics_div.find_all('div', class_='caption')
            performance_values.append(captions[0].text.strip())
            response_times.append(captions[1].text.strip())
            completion_rates.append(captions[2].text.strip())

    # Now you have extracted the name, request entry, and performance metrics for each API
    for i in range(len(api_names)):
        print(f"API Name: {api_names[i]}")
        print(f"Request Entry: {api_requests[i]}")
        print(f"Performance Value: {performance_values[i]}")
        print(f"Response Time: {response_times[i]}")
        print(f"Completion Rate: {completion_rates[i]}")
        print("-" * 20)

else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
