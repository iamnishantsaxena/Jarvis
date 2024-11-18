import io
import requests, random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK packages
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

weather_api_key = "your_weather_api_goes_here"

ADDITIONAL_RESPONSES = {
    "how are you": [
        "I'm just a chatbot, but thanks for asking!",
        "I'm doing well! How about you?",
        "I'm a bot, so no complaints here, but I appreciate you asking!",
        "I'm functioning as expected, thank you!"
    ],
    "tell me a joke": [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why was the math book sad? Because it had too many problems.",
        "What do you call fake spaghetti? An impasta!",
        "Why don’t skeletons fight each other? They don’t have the guts."
    ],
    "who created you": [
        "Just one amazing person – Nishant! A one-man army built me from scratch.",
        "It was Nishant, the wizard of code, who single-handedly brought me to life!",
        "I'm the brainchild of Nishant – no team, just pure solo genius!",
        "Nishant is my creator, coder, and all-around chatbot superhero!"
    ],

    "bye": [
        "Goodbye! Feel free to come back if you have more questions.",
        "Catch you later! Don't hesitate to return if you need anything else.",
        "See you soon! Have a great day ahead!",
        "Take care! I'll be here whenever you need me."
    ],
    "what's your favorite color": [
        "I don't have personal preferences, but I can help with your questions.",
        "I don't see colors, but I do love helping you with any topic!",
        "Colors are fascinating, but I don’t have one to call my favorite.",
        "I don't have a favorite color, but tell me yours!"
    ],
    "who won the World Series in 2020": [
        "The Los Angeles Dodgers won the World Series in 2020.",
        "The Dodgers clinched the 2020 World Series title.",
        "In 2020, the Los Angeles Dodgers emerged victorious in the World Series."
    ],
    "recommend a book": [
        "It depends on your interests. Do you prefer fiction or non-fiction?",
        "I'd suggest 'The Great Gatsby' if you're into classics. How about you?",
        "How about 'Sapiens' by Yuval Noah Harari? It's a fascinating read.",
        "If you like thrillers, 'The Silent Patient' by Alex Michaelides might be a great choice."
    ],
    "what's the meaning of life": [
        "The meaning of life is a profound and philosophical question. It varies from person to person.",
        "Some say it’s to seek happiness, others believe it’s about personal growth. What do you think?",
        "The meaning of life is subjective and can be different for everyone. It's up to you to define it!",
        "The meaning of life? A deep question, often shaped by personal experience and perspective."
    ],
    "how can I learn programming": [
        "You can start by learning a programming language like Python and practice regularly.",
        "Begin with Python! It’s beginner-friendly and widely used in the tech industry.",
        "I'd recommend starting with Python. It's powerful, simple, and versatile.",
        "To learn programming, start by understanding basic concepts and pick a language like Python or JavaScript."
    ],
     "what's the weather like today": [
        "I don't have access to real-time data, but you can check a weather website or app.",
        "Sorry, I can't provide live weather updates. Try your local weather app.",
        "For the latest weather, you might want to check a weather website or app.",
        "Unfortunately, I can't access weather updates. Please check your favorite weather service."
    ],
    "what's your favorite movie": [
        "I don't watch movies, but I can discuss movie recommendations.",
        "I don’t watch movies, but I’m happy to recommend some based on your preferences!",
        "I can’t pick a favorite, but I can talk about popular movies! Any genre you like?",
        "Movies? I don't watch them, but I know a lot about them. What’s your favorite?"
    ],
    "where is the Eiffel Tower located": [
        "The Eiffel Tower is located in Paris, France.",
        "In Paris, France, the Eiffel Tower stands tall.",
        "The Eiffel Tower can be found in the beautiful city of Paris, France."
    ],
    "tell me about artificial intelligence": [
        "Artificial intelligence (AI) is the simulation of human intelligence by machines.",
        "AI enables machines to learn from experience, adapt to new inputs, and perform human-like tasks.",
        "Artificial intelligence refers to machines that can perform tasks that typically require human intelligence."
    ],
    "who is your favorite celebrity": [
        "I don't have preferences, but I can provide information about various celebrities.",
        "I don’t follow celebrities, but I can talk about them. Who's your favorite?",
        "I don’t have personal preferences, but I know a lot about celebrities if you'd like to talk about them."
    ],
    "what's the capital of Japan": [
        "The capital of Japan is Tokyo.",
        "Tokyo is the capital city of Japan.",
        "The bustling capital of Japan is Tokyo."
    ],
    "how does a computer work": [
        "A computer processes data using a combination of hardware and software.",
        "Computers work by processing data through circuits, memory, and software that executes instructions.",
        "At a basic level, a computer takes input, processes it, and produces output using software and hardware."
    ],
    "do you like pizza": [
        "I can't eat, but I can help you find pizza places near you.",
        "I don't have taste buds, but pizza sounds delicious! What toppings do you like?",
        "Though I can't enjoy pizza, I can certainly help you order some!"
    ],
    "tell me a fun fact": [
        "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.",
        "Here's a fun fact: The shortest war in history lasted just 38-45 minutes, between Britain and Zanzibar in 1896.",
        "Here’s one: Octopuses have three hearts!"
    ],
    "what's the largest planet in our solar system": [
        "Jupiter is the largest planet in our solar system.",
        "The largest planet in our solar system is Jupiter, known for its Great Red Spot.",
        "Jupiter holds the title of being the largest planet in our solar system."
    ],
    "how does a search engine work": [
        "Search engines use web crawlers to index websites and algorithms to rank and display search results.",
        "A search engine finds websites by crawling the internet and then ranks them based on relevance to your query.",
        "Search engines index websites and rank them using complex algorithms to give you the best results."
    ],
    "tell me a riddle": [
        "I'm in the middle of water but never get wet. What am I? Answer: A shadow.",
        "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I? Answer: An echo.",
        "What has keys but can't open locks? Answer: A piano."
    ],
    "what's the population of India": [
        "As of my last update in 2021, India's population is over 1.3 billion people.",
        "India's population is currently over 1.3 billion people, making it the second most populous country.",
        "India has a population exceeding 1.3 billion, as of 2021."
    ],
    "what's the square root of 144": [
        "The square root of 144 is 12.",
        "The square root of 144 is 12, a perfect square.",
        "12 is the square root of 144."
    ],
    "recommend a TV show": [
        "What genre are you interested in? Comedy, drama, science fiction, or something else?",
        "I can recommend shows like 'Breaking Bad', 'Stranger Things', or 'The Office'. What are you in the mood for?",
        "Looking for a TV show recommendation? How about 'The Mandalorian' or 'Black Mirror'?"
    ],
    "what's the difference between HTML and CSS": [
        "HTML is used for structuring web content, while CSS is used for styling and layout.",
        "HTML creates the structure of a webpage, and CSS is used to design its look.",
        "HTML builds the skeleton of a webpage, and CSS dresses it up with colors and layout."
    ],

    "how can I learn a new language": [
    "You can start by practicing with apps like Duolingo or Memrise, and immerse yourself in the language through media.",
    "To learn a new language, try starting with common phrases and then build your vocabulary and grammar.",
    "Start by learning basic phrases, practicing daily, and using language learning apps or classes."
    ],

    "what's your favorite hobby": [
        "I don't have hobbies, but I can help you find some interesting ones to try!",
        "I don’t have hobbies, but I’m here to help you explore new ones!",
        "I don’t have hobbies, but I know a lot about them! What are your hobbies?"
    ],

    "how can I improve my coding skills": [
        "You can improve by practicing regularly, working on projects, and learning from online tutorials.",
        "To get better at coding, focus on problem-solving and build small projects to apply your knowledge.",
        "Keep learning new concepts, practice daily, and work on building projects to strengthen your coding skills."
    ],

    "tell me a fun fact about animals": [
        "Did you know that octopuses have three hearts and blue blood?",
        "A group of flamingos is called a 'flamboyance'!",
        "An elephant’s brain weighs about 5 kg, which is 1/10th of its body weight!"
    ],

    "how can I stay motivated": [
        "Set clear goals, break them down into smaller tasks, and track your progress.",
        "To stay motivated, find a purpose behind your tasks and remind yourself of your goals regularly.",
        "You can stay motivated by focusing on small wins and celebrating your achievements along the way."
    ],

    "who was Albert Einstein": [
        "Albert Einstein was a theoretical physicist known for developing the theory of relativity.",
        "Albert Einstein is famous for his theory of relativity and his equation, E=mc^2.",
        "Albert Einstein, one of the most influential scientists, revolutionized our understanding of space, time, and energy."
    ],

    "how do I relax": [
        "Try meditation, deep breathing exercises, or listening to calming music to relax.",
        "Taking a walk, practicing mindfulness, or reading a book can also help you relax.",
        "Relaxing can be as simple as stretching, practicing breathing exercises, or just taking a break from screens."
    ],

    "tell me a riddle": [
        "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I? Answer: An echo.",
        "I have keys but open no locks. What am I? Answer: A piano.",
        "The more of this there is, the less you see. What is it? Answer: Darkness."
    ],

    "how can I get better at public speaking": [
        "Practice in front of a mirror, join a speaking club, and work on your confidence.",
        "To improve, try recording yourself, watching your performances, and focusing on body language.",
        "Start small, practice in front of friends, and always be mindful of your tone and posture."
    ],

    "what's the best way to start my day": [
        "Start your day with a positive mindset, a healthy breakfast, and a plan for the day.",
        "A great morning routine includes stretching, drinking water, and setting clear goals for the day.",
        "Consider a quick workout, a healthy breakfast, and setting intentions for your day to start right."
    ],

    "how do I deal with failure": [
        "View failure as an opportunity to learn and grow. Reflect on what went wrong and try again.",
        "Failure is part of the journey to success. Learn from it and move forward with a new approach.",
        "The key to handling failure is resilience—embrace the lessons and use them to fuel your next attempt."
    ],

    "tell me a space fact": [
        "Space is completely silent because there is no air to carry sound waves.",
        "There are more stars in the universe than grains of sand on all the Earth’s beaches.",
        "Did you know that one day, the Sun will expand into a red giant and possibly engulf Earth?"
    ],

    "how can I be more productive": [
        "Try using techniques like the Pomodoro method, setting clear goals, and eliminating distractions.",
        "To increase productivity, break your tasks into smaller, manageable chunks and prioritize them.",
        "Staying organized and taking regular breaks can significantly boost your productivity."
    ],

    "how do I make new friends": [
        "Start by joining social groups or activities that interest you and engage with people there.",
        "Be open, approachable, and take an interest in others. Sometimes small talk leads to lasting friendships.",
        "Join clubs or communities where you can meet like-minded people and connect over shared interests."
    ],

    "what's the best way to study": [
        "Find a quiet, comfortable space, break your study sessions into manageable chunks, and take regular breaks.",
        "Use active learning methods like summarizing, teaching others, and testing yourself for better retention.",
        "Create a study plan, avoid distractions, and make sure to review regularly to improve your understanding."
    ],

    "how can I develop a growth mindset": [
        "Start by embracing challenges, learning from mistakes, and focusing on effort rather than results.",
        "A growth mindset is about seeing failure as a learning opportunity and constantly striving to improve.",
        "To cultivate a growth mindset, focus on perseverance, learning, and embracing challenges."
    ],

    "what's the meaning of happiness": [
        "Happiness is a state of mind, often linked to fulfillment, gratitude, and emotional well-being.",
        "For many, happiness is found in meaningful connections, personal achievements, and inner peace.",
        "The meaning of happiness is subjective, but it often revolves around contentment, love, and purpose."
    ],

    "how do I stay organized": [
        "Use tools like calendars, task lists, and reminders to keep everything in order.",
        "Start by decluttering your workspace, and break big tasks into smaller, more manageable ones.",
        "Staying organized involves consistency—create a routine and stick to it to maintain order."
    ],

    "tell me about famous composers": [
        "Famous composers include Ludwig van Beethoven, Wolfgang Amadeus Mozart, and Johann Sebastian Bach.",
        "Some of the most well-known composers are Beethoven, Mozart, and Tchaikovsky.",
        "Beethoven, Mozart, and Bach are legendary composers whose music is timeless and influential."
    ],

    "what's the best way to save money": [
        "Start by tracking your expenses, setting a budget, and cutting down on unnecessary purchases.",
        "Save a percentage of your income each month, and look for ways to reduce recurring costs.",
        "To save money, avoid impulse purchases and try using savings apps or programs."
    ],

    "how do I manage my emotions": [
        "Try mindfulness exercises, take deep breaths, and practice self-compassion to manage your emotions.",
        "Journaling, talking with friends, or seeking professional help can also help with emotional management.",
        "When feeling overwhelmed, take a moment to pause, reflect, and focus on solutions rather than emotions."
    ]
    
}

class Chatbot:
    def __init__(self, corpus_path):
        # Read and preprocess the corpus
        with open(corpus_path, 'r', encoding='utf8', errors='ignore') as fin:
            self.raw = fin.read().lower()
        
        self.sent_tokens = nltk.sent_tokenize(self.raw)
        self.word_tokens = nltk.word_tokenize(self.raw)
        self.lemmer = WordNetLemmatizer()
        self.sent_tokens.pop(-1)

        self.greeting_inputs = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
        self.greeting_responses = ["hi", "hey", "*nods*", "hi there", "hello", "I'm glad you're talking to me!"]
        self.additional_responses = ADDITIONAL_RESPONSES
    
    def preprocess_input(self, text):
      remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
      return text.lower().translate(remove_punct_dict)
    
    def LemTokens(self, tokens):
        return [self.lemmer.lemmatize(token) for token in tokens]

    def LemNormalize(self, text):
        return self.LemTokens(nltk.word_tokenize( self.preprocess_input(text) ))

    def greeting(self, sentence):
        for word in sentence.split():
            if word.lower() in self.greeting_inputs:
                return random.choice(self.greeting_responses)
        return None

    def generate_response(self, user_response):
        robo_response = ''
        self.sent_tokens.append(user_response)

        TfidfVec = TfidfVectorizer(tokenizer=self.LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(self.sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        if req_tfidf == 0:
            robo_response = "I am sorry, I don't understand you."
        else:
            r_response = self.sent_tokens[idx]
            robo_response = r_response.split("jarvis:")[-1].strip()
        
        # Check for predefined additional responses
        if user_response in self.additional_responses:
            responses = self.additional_responses[user_response]
            robo_response = random.choice(responses)

        return robo_response

    def chat(self):
        print("Jarvis: My name is Jarvis. I'm here to answer your questions. Type 'bye' to exit.")
        flag = True
        while flag:
            user_response = input("You: ").lower()
            user_response = self.preprocess_input(user_response)
            if not user_response.strip():
                print("Jarvis: Please enter a valid question or statement.")
                continue
            if user_response == 'bye':
                flag = False
                print("Jarvis: Goodbye! Feel free to chat again anytime!")
            elif user_response in ('thanks', 'thank you'):
                flag = False
                print("Jarvis: You're welcome! Have a great day.")
            elif weather_api_key!="your_weather_api_goes_here" and "weather" in user_response:
                print("Sure! Let me fetch the weather details for you.")
                city = user_response.split("in")[-1].strip()
                # Ensure the city name is not empty
                if city:
                    weather_info = weather_api.get_weather(city)
                    print(f"Jarvis: {weather_info}")
                else:
                    print("Jarvis: I couldn't understand the city name. Could you please provide it after 'in'?")
            else:
                greeting_response = self.greeting(user_response)
                if greeting_response:
                    print(f"Jarvis: {greeting_response}")
                else:
                    response = self.generate_response(user_response)
                    if response:
                      print(f"Jarvis: {response}")
                    
                    
class WeatherAPI:
    def __init__(self, weather_api_key):
        self.weather_api_key = weather_api_key

    def get_weather(self, query_param):
        api_host = "open-weather13.p.rapidapi.com"
        base_url = f"https://open-weather13.p.rapidapi.com/city/{query_param}/EN"
        
        headers = {
            "X-RapidAPI-Key": self.weather_api_key,
            "X-RapidAPI-Host": api_host
        }

        try:
            # Making the request to the API
            response = requests.get(base_url, headers=headers)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code != 200:
                return f"Error: Unable to fetch data. Status code: {response.status_code}"

            data = response.json()

            # Check if the response contains error information
            if 'error' in data:
                return f"Error: {data['error']['message']}"

            # Extracting relevant weather data
            location = data.get('location', {}).get('name', 'Unknown')
            country = data.get('location', {}).get('country', 'Unknown')
            condition_text = data.get('current', {}).get('condition', {}).get('text', 'No data')
            temperature_c = data.get('current', {}).get('temp_c', 'No data')
            humidity = data.get('current', {}).get('humidity', 'No data')

            # Return the weather summary
            return f"The weather in {location}, {country} is {condition_text}. " \
                   f"The temperature is {temperature_c}°C, and the humidity is {humidity}%. "

        except requests.RequestException as e:
            return f"Error fetching weather information: {e}"  

# Main Execution
if __name__ == "__main__":
    weather_api = WeatherAPI(weather_api_key)

    chatbot = Chatbot('jarvis.txt')
    chatbot.chat()