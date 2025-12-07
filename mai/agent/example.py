"""
Example usage of the agentic reader.

This script demonstrates how to use the agentic reader to analyze
a latex document and answer questions about it.
"""

import asyncio
from mai.agent.agentic_reader import agentic_reader, agentic_reader_stream, AgenticReaderOptions


async def simple_example():
    """Simple example of using the agentic reader."""
    # Sample LaTeX content
    latex_content = r"""
\documentclass{article}
\usepackage{graphicx}

\section{Introduction to Machine Learning}

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

\section{Types of Machine Learning}

\subsection{Supervised Learning}
In supervised learning, the model learns from labeled training data.

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{training.png}
\caption{Training Process}
\label{fig:training}
\end{figure}

\subsection{Unsupervised Learning}
Unsupervised learning finds patterns in unlabeled data.

\section{Applications}

Machine learning is used in:
\begin{itemize}
\item Image recognition
\item Natural language processing
\item Recommendation systems
\end{itemize}
"""

    question = "What are the main types of machine learning mentioned in the document?"

    # Run the agentic reader
    result = await agentic_reader(
        question=question,
        text_content=latex_content,
        options=AgenticReaderOptions(
            max_iterations=5,
            model='openai:gpt-5-mini',
            include_metadata=True,
        )
    )

    print("Answer:", result.answer)
    if result.metadata:
        print("\nMetadata:", result.metadata)


async def streaming_example():
    """Example of using the agentic reader with streaming events."""
    latex_content = r"""
\documentclass{article}

\section{Python Programming}

Python is a high-level programming language known for its simplicity.

\subsection{Features}
\begin{itemize}
\item Easy to learn
\item Powerful libraries
\item Large community
\end{itemize}
"""

    question = "What are the key features of Python?"

    print("Streaming events:\n")
    async for event, data in agentic_reader_stream(
        question=question,
        text_content=latex_content,
        options=AgenticReaderOptions(max_iterations=5),
    ):
        print(f"Event: {event}")
        print(f"Data: {data}\n")



if __name__ == '__main__':
    print("=" * 60)
    print("Simple Example")
    print("=" * 60)
    asyncio.run(simple_example())
