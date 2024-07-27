# OpenAdvisor

OpenAdvisor is a machine learning-based stock recommendation application developed for the FossHack 2024 hackathon. It provides a self-hostable investment platform designed for common investors, offering an easy-to-use interface for stock predictions and portfolio management.

## Features

- Machine learning-based stock predictions using CatBoost
- Daily/weekly stock rankings and recommendations
- User-friendly dashboard for tracking investments
- Backtesting module with performance visualization
- Customizable prediction frequency settings
- Self-hostable platform

## Tech Stack

OpenAdvisor utilizes a modern and efficient tech stack, carefully chosen to provide a robust, scalable, and user-friendly platform. Here's a detailed breakdown of the technologies used and the reasoning behind each choice:

### Backend

- **Python**: A versatile and powerful programming language, ideal for data processing, machine learning, and web development.
  - **Flask**: A lightweight and flexible Python web framework. Flask's simplicity and extensibility make it perfect for building RESTful APIs and serving our web application.

### Frontend

- **Alpine.js**: A lightweight JavaScript framework for adding interactivity to web pages. Its simplicity and small footprint make it ideal for creating responsive user interfaces without the overhead of larger frameworks.
- **Daisy UI**: A plugin for Tailwind CSS that provides a set of pre-designed components. This allows for rapid UI development while maintaining a consistent and professional look.
- **Tailwind CSS**: A utility-first CSS framework that enables quick styling and responsive design. Tailwind's approach allows for highly customizable designs without writing custom CSS.

### Database

- **SQLite**: A lightweight, serverless database engine. SQLite is chosen for its simplicity in setup and maintenance, making it ideal for a self-hostable application. It's perfect for small to medium-scale deployments.

### Machine Learning

- **CatBoost**: An open-source gradient boosting library developed by Yandex. CatBoost is chosen for its high performance, ability to handle categorical features automatically, and resistance to overfitting. These qualities make it well-suited for stock price prediction tasks.

### Data Source

- **yfinance**: A Python library that provides a simple way to download historical market data from Yahoo Finance. It's reliable, easy to use, and provides access to a wide range of financial data needed for stock analysis and prediction.

### Development and Deployment Tools

- **pip**: The package installer for Python, used to manage project dependencies.
- **Node.js and npm**: Used for managing frontend dependencies and build processes.
- **Git**: For version control and collaborative development.

This tech stack is designed to provide a balance between performance, ease of development, and maintainability. It allows for rapid development and iteration, crucial for a hackathon project, while also providing a solid foundation for potential future scaling and enhancement.

## Implementation Steps
1. **Initial Setup**:
   - Set up the project repository, initialize the backend framework, and create the frontend structure.
   
2. **User Authentication**:
   - Implement user registration, login, and profile management.
   
3. **Historical Data Integration**:
   - Import and process historical stock data from public datasets.
   
4. **Recommendation Engine**:
   - Develop the AI model to analyze user data and historical trends for stock recommendations.
   
5. **Portfolio Management**:
   - Build features for users to add, track, and analyze their investments.

## Potential Challenges
- Ensuring the accuracy and reliability of the AI-driven recommendations.
- Efficiently processing and analyzing large volumes of historical data.
- Maintaining user data privacy and security.


This project adheres to the FOSS Hackathon rules and provides a valuable tool for personal and professional investors, leveraging open-source technologies and datasets.
