import datetime
from math import floor
import requests
from bs4 import BeautifulSoup
import pandas as pd


def calculate_time_gap(timestamp_string):
    timestamp = datetime.datetime.strptime(timestamp_string, "%Y-%m-%dT%H:%M:%S")
    current_time = datetime.datetime.now()
    time_gap = (current_time - timestamp).total_seconds() / 3600
    return floor(time_gap)


hackernews_file = 'hackernews_stories.csv'


def scrape_hackernews_stories():
    stories_data = []
    for page in range(1, 6):
        url = f"http://news.ycombinator.com/news?p={page}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        story_rows = soup.select(".athing")

        for row in story_rows:
            rank = row.find(class_="rank").text.strip(".")
            title = row.find(class_="titleline").text
            next_row = row.find_next_sibling("tr")

            if next_row:
                original_age_string = next_row.find(class_="age")['title']
                age = calculate_time_gap(original_age_string)
                points = int(next_row.find(class_="score").text.split()[0]) if next_row.find(class_="score") else 0

                comments_link = next_row.find("a", href=lambda href: href and "item?id" in href)
                if comments_link:
                    comments_text = comments_link.text.strip()
                    comments = int(comments_text.split()[0]) if comments_text.split()[0].isdigit() else 0
                else:
                    comments = 0
            else:
                age = 0
                points = 0
                comments = 0

            stories_data.append({
                "Rank": rank,
                "Title": title,
                "Age(hours)": age,
                "Points": points,
                "Comments": comments
            })

    df = pd.DataFrame(stories_data)
    df.to_csv(hackernews_file, index=False)
    print(f"Scraping completed. Data saved to {hackernews_file}.")


if __name__ == "__main__":
    scrape_hackernews_stories()