import os
from dotenv import load_dotenv

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# load environment variables
load_dotenv()

# openai key should be stored in .env as OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# initialize LLM once at startup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def ask_llm(user_input: str, expert: str) -> str:
    """送信されたテキストと専門家の選択に応じてLLMに問い合わせ、回答を返す。"""

    if expert == "技術コンサルタント":
        system_msg = (
            "あなたは技術コンサルタントの専門家として振る舞ってください。"
            "ユーザーの質問に技術的な観点から助言を行ってください。"
        )
    else:
        system_msg = (
            "あなたはマーケティングスペシャリストの専門家として振る舞ってください。"
            "ユーザーの要望にマーケティングの観点から回答してください。"
        )

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(messages)
    return response.content


def main():
    st.title("LangChain LLM 専門家デモアプリ")
    st.write(
        "このアプリは入力したテキストをLLMに渡し、" 
        "ラジオボタンで選択した専門家として振る舞わせます。"
        "テキストを入力し、専門家を選んでから「送信」してください。"
    )

    user_text = st.text_area("入力テキスト", height=150)
    expert = st.radio(
        "専門家を選択してください",
        ("技術コンサルタント", "マーケティングスペシャリスト"),
    )

    if st.button("送信"):
        if not user_text.strip():
            st.warning("テキストを入力してください。")
        else:
            with st.spinner("LLMからの回答を取得中..."):
                answer = ask_llm(user_text, expert)
            st.subheader("LLMの回答")
            st.write(answer)


if __name__ == "__main__":
    main()
