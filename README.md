# Movie-Media-Content-Moderation-using-LLM

**Live Demo Link:** https://movie-moderator.streamlit.app/

## Automated Content Moderation with Retrieval-Augmented Generation
This project aims to develop an AI agent for automated content moderation, enabling users to make informed decisions about the suitability of books and movies for different settings, especially for children. The system employs a hybrid AI architecture that combines Retrieval-Augmented Generation (RAG) with advanced natural language processing models to perform contextual text analysis and summarization.

## Research Question
This study addresses the need for automated content moderation in determining the appropriateness of books and movies by developing an AI agent for automated content moderation. The research investigates how a Retrieval-Augmented Generation (RAG) approach can effectively identify and summarize content that requires moderation, such as profanity, violence, and positive messages, to assist users in making informed decisions about its suitability for different settings, especially for children.

## Methodology
The project employs a hybrid AI architecture that combines Retrieval-Augmented Generation with advanced natural language processing models (GPT-3.5, GPT-4, and Sentence-Transformers all-MiniLM-L6-v2). The system performs efficient data handling and scalable querying of embedded text data. Text data are sourced from comprehensive corpora of book texts and movie subtitles, which undergo preprocessing to align with identified content categories.

## Contributions
This research introduces an application of the Retrieval-Augmented Generation (RAG) framework, utilizing it as a core methodology to significantly refine the process of automated content moderation. The RAG framework is applied to generate contextual responses to user queries, enhancing the precision and relevance of the information provided regarding content suitability. By integrating state-of-the-art transformer-based models, the study demonstrates an approach to content-sensitive querying. This not only supports the adaptability of the AI agent in real-time interactions but also optimizes the relevance of retrieved data, aligning closely with user-specific content moderation needs. This contribution marks a critical advancement in employing hybrid AI models to tackle complex, query-driven data retrieval and generation tasks in the domain of content moderation.

## Results
The AI system effectively retrieves and synthesizes relevant excerpts that address specific content concerns in books and movies. For instance, inquiries about "Harry Potter and the Philosopher's Stone" for a young audience precisely pull detailed sections discussing violence and highlight positive messages, tailored for easy user comprehension.

## Implications for the Field of AI
The integration of retrieval-augmented generation into content moderation showcases significant advancements in AI's capabilities to interpret and summarize complex multimedia content which are very specific and free from hallucinations. This approach underlines the potential for AI to support nuanced decision-making in content consumption, setting a foundation for future research in more refined content sensitivity and personalized information delivery. The study not only enhances understanding of hybrid AI model applications but also suggests a robust framework for real-world implementations across diverse digital platforms.

## Getting Started
To get started with this project, follow these steps:

1. _Navigate to work directory:_ Change directory into `streamlit_app` folder.
2. _Clone the repository:_ Run `git clone https://github.com/Zain-Saiyed/Media-Content-Moderation-using-LLM.git`
3. _Install the required dependencies:_ `pip install -r requirements.txt`
4. _Start the application:_ Run `streamlit run app.py`.

## Contributing
Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgements

Credits to OpenAI API for [GPT-3.5 Turbo](https://openai.com/api/), & [Sentence-Transformers all-MiniLM-L6-v2](https://www.sbert.net/docs/pretrained_models.html).
