# CourseBot
'The website offers a variety of online courses, including programming basics with Java, designing a chatbot similar to ChatGPT, web development, a Python project playground for kids, AI productivity hacks, HTML, CSS, and JavaScript essentials, game development using Python, cloud computing on Amazon AWS, and Python programming for beginners, intermediates, and advanced learners.'

inference code 
import requests

url = "http://localhost:5000/ask"

payload = {"question": "What technical courses are available?"}

headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(response.json())
