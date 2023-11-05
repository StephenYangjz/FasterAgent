import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import csv

# Initialize the WebDriver (make sure to specify the path to the downloaded WebDriver executable)
driver = webdriver.Chrome()

categories = [
    "Sports",
    "Finance",
    "Data",
    "Entertainment",
    "Travel",
    "Location",
    "Science",
    "Food",
    "Transportation",
    "Music",
    "Business",
    "Visual Recognition",
    "Tools",
    "Text Analysis",
    "Weather",
    "Gaming",
    "SMS",
    "Events",
    "Health and Fitness",
    "Payments",
    "Financial",
    "Translation",
    "Storage",
    "Logistics",
    "Database",
    "Search",
    "Reward",
    "Mapping",
    "Artificial Intelligence/Machine Learning",
    "Email",
    "News, Media",
    "Video, Images",
    "eCommerce",
    "Medical",
    "Devices",
    "Business Software",
    "Advertising",
    "Education",
    "Media",
    "Social",
    "Commerce",
    "Communication",
    "Other",
    "Monitoring",
    "Energy",
    "Jobs",
    "Movies",
    "Cryptography",
    "Cybersecurity"
]

# Define the URL of the RapidAPI Sports category page


# driver.get("https://rapidapi.com/category/Sports")

# headless
options = webdriver.ChromeOptions()
options.add_argument('headless')
driver = webdriver.Chrome(options=options)
# no pic

# # Find and click the "Next Page" button (replace with the actual selector of the button)
# next_page_button = driver.find_element(By.CSS_SELECTOR, ".next-page-button")
# next_page_button.click()

# Send an HTTP GET request to the URL
# response = requests.get(url)
for cat in categories:
    print(cat)
    url = f"https://rapidapi.com/category/{cat}"
    driver.get(url)
    # Check if the request was successful (status code 200)
    while True:
        try:
            from time import sleep
            sleep(0.5)
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # You can now work with the parsed HTML content using BeautifulSoup
            # For example, let's extract and print the title of the page:
            title = soup.title.string
            # print(f"Page Title: {title}")

                # You can also extract other information from the page as needed

            # Find all the div elements with class 'ItemCard'
            item_cards = soup.find_all('div', class_='ItemCard')

            # Initialize empty lists to store the extracted information
            api_names = []
            api_requests = []
            performance_values = []
            response_times = []
            completion_rates = []

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

                # Initialize placeholders for captions
                performance_value = 'N/A'
                response_time = 'N/A'
                completion_rate = 'N/A'

                # Iterate through the 'Metrics' divs within the current 'ItemCard'
                for metrics_div in metrics_divs:
                    captions = metrics_div.find_all('div', class_='caption')
                    
                    if captions:
                        # Extract and update the values based on the order in the HTML
                        for i, caption in enumerate(captions):
                            caption_text = caption.text.strip()
                            if i == 0:
                                performance_value = caption_text
                            elif i == 1:
                                response_time = caption_text
                            elif i == 2:
                                completion_rate = caption_text

                # Append the data to lists
                performance_values.append(performance_value)
                response_times.append(response_time)
                completion_rates.append(completion_rate)


            # Specify the existing CSV file path where you want to add the data
            csv_file_path = 'api_data.csv'

            # Open the existing CSV file in append mode and create a CSV writer
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)

                # Write data rows to the CSV file
                for i in range(len(api_names)):
                    api_name = api_names[i]
                    api_request = api_requests[i]
                    performance_value = performance_values[i]
                    response_time = response_times[i]
                    completion_rate = completion_rates[i]

                    # Write the data for the current API to the CSV file
                    csv_writer.writerow([cat, api_name, api_request, performance_value, response_time, completion_rate])


            # Now you have extracted the name, request entry, and performance metrics for each API
            # for i in range(len(api_names)):
            #     print(f"API Name: {api_names[i]}")
            #     print(f"Request Entry: {api_requests[i]}")
            #     print(f"Performance Value: {performance_values[i]}")
            #     print(f"Response Time: {response_times[i]}")
            #     print(f"Completion Rate: {completion_rates[i]}")
            #     print("-" * 20)
            next_page_button = driver.find_element(By.XPATH, "//button[text()='›']")
            next_page_button.click()
        except Exception as e:
            # If the "›" button is not found, exit the loop
            break
# Don't forget to close the WebDriver when you're done
driver.quit()