Stasiu Tippett

Computer Science 311

February 27, 2026

Introduction

The goal of this project was to build a FAQ chatbot that can answer
questions regarding a product. At first, This project uses some cheap
earbuds that are nice but a little bit hard to use. They are a good
candidate for this project because they come from a less common
manufacturer so looking up how to use them can be more difficult.
Instead of reading a long annoying FAQ or looking at google or large
language model (LLM) output without context this project allows the user
to simply chat with an LLM that understands the parameters and steps
outlined in the FAQ. The process of giving AI context of data is called
Retrieval Augmented Generation (RAG). When using an LLM the model has
training information, but that information could be incomplete or
inaccurate. Additionally, LLM's are known to hallucinate i.e. provide a
confident answer that is essentially made up. RAG systems allow the LLM
to reference accurate and up to date documentation. The question can be
made in natural language instead of looking through documentation and
crafting search terms. The chat bot has a CLI tool and a graphical UI
for whichever the user prefers.

Methods used

RAG sounds good but how does it work under the hood. The RAG pipeline
used in this project is indicative of the standard application that uses
RAG. The base of the pipeline is the document being stored in a vector
database. Next the applicable pieces of information are retrieved from
the database using the semantics of the question. The pieces of
information are then passed to the LLM API. The model itself is
stateless so being given data which has been pulled from a database
ideally grounds the LLM in objective references. This can minimize or
remove the hallucination problem which plagues LLM.

What is RAG? Rag has two primary components. The ability to search
through a persistent source of information. In the case of this project
the FAQ documents. The second component is the generative model (LLM)
itself. RAG integrates the retrieval into the generative process itself.
But why is it necessary? LLMs commonly produce confident and even
seemingly coherent answers which turn out to be wrong in the end. This
becomes more of a problem when the answer is on a topic not inside the
models training. The model might not contain the data for several
reasons. Maybe the data came out after the training was completed or
maybe the data is private, and the RAG pipeline accesses data which
belongs to an individual or organization. Or it could be an obscure
topic that the model only vaguely understands. RAG solves for this by
having verifiable context taken out of the source docs if it is relevant
and has the needed information the LLM output would be vastly more
accurate. This has a grounding effect on the LLMs output.

Text Embedding what is it and how does it work? The next method used is
called text embedding. We remember what a vector is from physics class
direction and magnitude as opposed to simple magnitude of scalars. Text
embedding takes small pieces of text and creates relations between them.
These relationships are stored in arrays and are called vectors because
they point in the direction of the other part of text and they have a
value that quantifies the distance between the two pieces of text. It is
essentially a mapping of semantic or fixed length chunks of text and
embedding into a high-dimensional vector space. This ties around to the
retrieval process. In order to retrieve correlated pieces of data,
knowing what is related to what is and how related they are is very
powerful. This allow the retrieval prosses to compare many small chunks
of text and measure by how related to each other they are enabling the
retravel of data that relates to a common topic. The users question
literally relates to information in the database which is bundled with
the question and passed to the API giving the model the ability to have
much more insight into the answer. Text Embedding works similarly to
LLMs in the sense that an embedding model is needed. A helpful way to
think of it like a reverser LLM although that is not entirely accurate
and more of an analogy. Conveniently the embedding models don't need as
many resources to get quite good results. For this project
all-MiniLM-L6-v2 was chosen. It is a good model for a project of this
scale because it is lightweight and fast. It is very supported with
LangChain libraries and addons to make it convenient to tie into the
project. This model is suited for data that is broken down at the
sentence level.

Large Language Models (LLMs) are a vital and the most hyped-up part of
the pipeline. Although the LLM is not the primary store of information
it plays an important part in synthesizing the final output. The LLM has
three tasks to complete. First it takes the retrieved chunks and reads
them. Then it reads and follows the actual prompt from the user.
Finally, it puts together the final output for the user including proper
formatting and style. Dividing up tasks is the real power of RAG. It was
particularly interesting in this implementation that the motel was run
locally on the same device using Ollama. It is amazing that 16GB of ram
is sufficient to use this kind of technology. The API exposed by the
local model is totally interchangeable with the more standard ones
accessible online. This is a benefit of the project because it could
work with local or over the internet based APIs.

The final method used in this project is text chunking. Text chunking is
needed because plain text can't be directly fed into the embedding
system. The embedding model is only able to handle chunks of text about
the size of a sentence. If too much text is sent to the embedding model
it will truncate the text and it will not be included. Missing text
would make retrieval less accurate. Breaking up a text file into smaller
chunks will function better with the input limits. In addition the size
of the pieces of information that can be accessed are much smaller
making it granular. Also, if the chunks of text are too big different
topics will be included in each chunk which will complicate things. The
chunks of text do have to align with the sentences and paragraphs of the
original text. This keeps the chunks much more manageable and logically
consistent. Like the LLM the embeddings library is run on the local
machine making things self-contained. In the case of the embeddings
model there is no API instead a python library interacts with the model.

Dataset Information

The source of the persistent data used in the RAG pipeline was the
manufacturer's website. The website had a
[FAQ](https://www.mi.com/global/support/faq/details/KA-509528/) that was
used as the data. To count the number of records simply counted the
number of chunks in the ChromaDB once the embedding was completed. The
dataset contained a total of 267 chunks each being its own record.

Prior to embedding the data was text. The full text counts as the
feature by itself. After the embedding was completed each record had
three features. The first feature is the text content itself this is a
chunk of text that was chunked by the embeddings model and is of string
data type. The second is the Embedding vector. The embedding vector is
an array of floats and is simply a multi dimensional vector array that
holds the vectors. The final feature is the chunk metadata. The metadata
is of Dictionary data type and is simply a python Dictionary. It holds
the position of that chunk in the source document and its location in
the ChromaDB itself.

Preprocessing is the process of preparing data for the embedding. The
first step in preprocessing is text sanitization. This was simply
accomplished by calling the LangChain TextLoader function and limiting
the encoding to utf-8. Next the text is chunked by passing it to
RecursiveCharacterTextSplitter function belonging to the LangChain text
splitter. It is passed two values 500 and 50. 500 represents the maximum
size of each chunk and 50 represents the allowable overlap. The overlap
allows for context to be kept for each chunk. The size is measured in
charictures in this case utf-8. Lastly the embedding model converts each
chunk to a vector and puts it in the ChromaDB. This allows the data to
persist with its meta data and relative position in the form of a
vector.

Libraries, Tools, and Frameworks

Like meany Python projects this one required a lot of dependencies. From
the vector embedding to the final UI these are the tools that made it
all possible.

- LangChain

- ChromaDB

- HuggingFace

- OpenAI Python client

- Streamlit

LangChain is a LLM orchestration framework that helps manage interacting
with multiple moving parts instead of a primitive chat with the LLM. It
can facilitate passing information to the LLM as well as allowing it to
interact with tools and things like that. This project only scratched
the surface of its capabilities. However model context protocol is a
more contemporary alternative for many of LangChain's capabilities.

ChromaDB is a persistent vector store for keeping and searching
vectorized chunks of data. Langchain can interact with it and make
passing vector data to the LLM a lot more convenient. The embedding
model will insert the data into the database, so the embedding model
only needs to run once. Alternatively the embedding model can keep the
embeddings in memory if referencing only needs to happen once.

HuggingFace is a website that is very important in the world of data
siance and machine learning. Especially in the world of open source
models and data. In this project the HuggingFace embedding model was
used to vectorize the data. The model was all-MiniLM-L6-v2 which was a
good size to run on a laptop. HuggingFace has an abundant and
overwhelming number of models and data that can be accessed by the open
source community.

OpenAI Python client is a wrapper that allows python to interact with
LLM APIs. OpenAI the creators of ChatGPT created the OpenAI API and it
became the current standard for LLM APIs. This python library allows
python programs to interact with LLMs over the internet via the API.
It's a convenient library to use because it only takes one function call
assigned to a variable and the program can start piping data to the LLM
and getting responses.

Lastly Streamlit is a web app UI library for a LLM chat interface. It
should not be used in production applications but for projects like this
it's a spectacular tool to spin up a quick UI to interact with LLMs. It
really was a pleasure to use because it only took a few lines of code to
get a working UI and a few small modifications to the TOML config file
and the interface was exactly how it should be for ease of use.

Application Design

The application design is the strong point of this project. The design
is quite clean and modular yet it gets a lot done. At a high level it
represents a flow of data from the boring FAQ to the structured,
formatted and hype-specific LLM output. The architecture flows from the
original text FAQ through sanitization which removes characters that
can't be chunked and stored in the database. Next it and its embeddings
are stored in the ChromaDB. Once the user presents a query it is
compared to the embeddings in the database to look for similarity. The
original question and any relevant data is passed to the LLM in
formatted manner so it can tell the question from the context. And
finally the LLM's response is received and passed to the user as a final
output. This is how the data flow through the pipeline to facilitate
RAG.

![](.\assignment1\media/media/image1.png){width="5.575in"
height="5.412239720034996in"}

The code is broken up into four files that each accomplish different
steps in the process. First is the embeddings.py which contains two
functions. One to sanitize chunk and embed the data. And the second to
query the database with the user's question. The llm.py file handles
calling the openAI API and passing it the user's data. It contains one
function called llm() that facilitates this. It includes the system
prompt that tells the LLM that it needs to respond and to keep responses
short. Thes functions are called from the app.py and the main.py. These
allow either the UI or the CLI interface depending on the user's
preference. Both main.py and app.py check if the database exists and if
not they create it using the provided data. The project contains a fair
amount of configuration that is handled by the various config files.
This includes the .env and the .streamlit files.

Instructions for Running

Getting this project up and running is pretty straightforward. First
navigate to the
[GitHub](https://github.com/marvelousMarsupial/CS311/tree/main/assignment1)
for the project then clone the repo to your machine. Next create a
Python environment and activate it. The project includes a
requirements.txt so simply use pip to install dependencies into the
local environment. Next obtain an LLM API key. For this project a local
LLM was used that is capable of running on a laptop. Its API key was
saved but any OpenAI compatible API key will work. The interface for
configuring this project is the .env file. Create one and populate with
the following values.

- DATA_PATH: file path to .txt file

- OPENAI_BASE_URL: URL the LLM is at.

- OPENAI_API_KEY: API key to preferred model

- MODEL_NAME: name is required for compatibility

- EMBEDDING_MODEL: The embedding model

- CHROMA_PATH: File path to the vector database

![](.\assignment1\media/media/image2.png){width="6.250872703412074in"
height="3.2817082239720037in"}

Here is what the configuration looked like. Keep in mind it's not a one
size fits all. To begin with only the LLM info needs to be provided
unless the user wants a custom implementation. The rest will fill itself
in if nothing custom is selected. The project is technically modular so
it could be pointed at a different text file of data. Also, a different
embeddings model could be used if preferred.

The project has two user interfaces a CLI and a UI. Simply run the
main.py to use the cli interface to talk to the model or run "streamlit
run app.py" to use the web app version. When using the UI version, the
command will open the browser, and a chat UI will be visible. If the
browser doesn't launch, simply open it and navigate to the URL displayed
in the terminal. If the cli is preferred just run the main.py file and
chat right in the terminal. The user can then ask the chat bot questions
pertaining to the product FAQ.

Results

Given the relatively low intelligence of the local models the output was
quite good and would work even better with a larger model like ChatGPT
or Claude. The super prompt had to be adjusted because at first the LLM
was giving long responses. Once told to keep the responses to one or two
sentences it worked well. The web UI also had to be configured in its
configuration file to make it more user friendly and minimalist. Once
set up it works well for answering questions about specific information
that is as up to date as the file and as specialized as needed. As
demonstrated in the screenshot the model is able to see numbers in the
FAQ like the battery life. Additionally it is able to semantically look
up information like how to pause. The noise cancellation question,
however, seems to not reference the FAQ and the LLM answered based off
its knowledge, which is important to note the limits of this kind of
technology.

![](.\assignment1\media/media/image3.png){width="6.5in"
height="3.254166666666667in"}

Insights

During this project a significant amount of insight was derived. Some
portions of the project ran smoothly but others had some friction. The
RAG responses were pretty grounded in the FAQ. It is visible in the
screenshot and can be compared to the FAQ in the GitHub. The local
Ollama LLM ran its inference without any problems and the output was
fairly good for its size. The modular structure of the project was also
a strong point. The separation of concerns with the different Python
files makes it manageable. And the modular setup allows for future
expansion or changes. Some of the friction point on the other hand made
the project less than perfect. First concern is the dataset. The dataset
is plain text instead of tables or some kind of structured data. There
is also a trade off with the sizes of the chunks of text. If the chunks
are too small context is lost but if the chunks of text are too large
the information becomes noisy. The shortcoming of the CLI interface is
it doesn't maintain a conversation memory. That problem is solved by the
UI. Lastly for production or serious use a more powerful model would be
a better fit. The local LLM is perfect for building the RAG pipeline,
but it doesn't have the most powerful intelligence.

There are some potential improvements that could be made on the project.
First a different data source could be used instead. It could be a CSV
or some other kind of more structured data as opposed to text. This
could facilitate filtered retrieval. Next the search of the data could
be multi-faceted i.e. it could search for keywords as well as the
semantic search. This would be an interesting approach to make the
gathering of data more thorough. A more coding heavy improvement would
be adding memory to the CLI interface so the LLM could see the
conversation history. Although there could be improvements the final
project is comprehensive and allows real use for talking to a FAQ.

References

Study.com. (n.d.). Practical application for artificial intelligence:
Learning chatbot. Retrieved February 24, 2026, from
<https://study.com/academy/lesson/practical-application-for-artificial-intelligence-learning-chatbot.html>

Study.com. (n.d.). Large language models and artificial intelligence.
Retrieved February 24, 2026, from
<https://study.com/academy/lesson/large-language-models-and-artificial-intelligence.html>

Xiaomi. (n.d.). Redmi Buds 6 FAQ. Retrieved February 24, 2026, from
https://www.mi.com/global/support/faq/details/KA-509528/
