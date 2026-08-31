"""
Docling Document Analysis Agent -- Streamlit App
=================================================
Optimized approach:
  - Document parsed ONCE and cached (no re-parse on re-run)
  - NO startup LLM calls (tools are lazy, agent calls them only when needed)
  - Clean Streamlit chat UI with file upload + URL input
  - Modern LangGraph create_react_agent
"""

import os
import streamlit as st
from docling.document_converter import DocumentConverter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "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"
)
DEFAULT_SOURCE = "https://arxiv.org/pdf/2408.09869"

SYSTEM_PROMPT = """You are an expert document analysis assistant powered by Docling.
Use tools selectively and efficiently:
- Call get_document_metadata first for orientation questions.
- Use search_pages_for_keyword to locate relevant pages before reading them.
- Prefer get_page_markdown over get_full_markdown for targeted questions.
- Always cite the page number when referring to specific content."""


# ------------------------------------------------------------------
# Tool Factory -- bound to the parsed Docling document
# ------------------------------------------------------------------
def build_tools(document):
    """Return a list of LangChain tools that close over the parsed document."""

    @tool
    def get_page_count() -> int:
        """Returns the total number of pages in the document."""
        return document.num_pages()

    @tool
    def get_document_metadata() -> str:
        """
        Returns structural metadata: page count, image count, table count,
        and the first 20 headings. Use this first for high-level questions.
        """
        num_pages  = document.num_pages()
        num_pics   = len(document.pictures)
        num_tables = len(document.tables)
        headings = []
        for item, _ in document.iterate_items():
            label = str(getattr(item, "label", ""))
            if label in ("section_header", "title") and hasattr(item, "text"):
                headings.append(item.text)
        heading_str = "\n".join(f"  - {h}" for h in headings[:20]) or "  (none detected)"
        return (
            f"Pages  : {num_pages}\n"
            f"Images : {num_pics}\n"
            f"Tables : {num_tables}\n"
            f"Headings (first 20):\n{heading_str}"
        )

    @tool
    def get_page_markdown(page_no: int) -> str:
        """
        Returns markdown content of a single page.
        Args:
            page_no: 1-indexed page number.
        """
        total = document.num_pages()
        if page_no < 1 or page_no > total:
            return f"Invalid page number {page_no}. Document has {total} pages."
        return document.export_to_markdown(page_no=page_no)

    @tool
    def get_full_markdown() -> str:
        """
        Returns the ENTIRE document as markdown.
        Use only when a broad question requires the whole document.
        Prefer get_page_markdown for targeted questions.
        """
        return document.export_to_markdown()

    @tool
    def get_tables(max_tables: int = 3) -> str:
        """
        Returns table content from the document.
        Args:
            max_tables: Maximum number of tables to return (default 3).
        """
        tables = document.tables
        if not tables:
            return "No tables found in this document."
        results = []
        for i, tbl in enumerate(tables[:max_tables]):
            try:
                df = tbl.export_to_dataframe()
                results.append(f"--- Table {i+1} ---\n{df.to_string(index=False)}")
            except Exception as e:
                results.append(f"--- Table {i+1} --- (parse error: {e})")
        return "\n\n".join(results)

    @tool
    def search_pages_for_keyword(keyword: str) -> str:
        """
        Finds pages containing a keyword and returns short excerpts.
        Use this to locate relevant pages before reading them in full.
        Args:
            keyword: Word or phrase to search (case-insensitive).
        """
        kw = keyword.lower()
        matches = []
        for page_no in range(1, document.num_pages() + 1):
            page_md = document.export_to_markdown(page_no=page_no)
            if kw in page_md.lower():
                idx     = page_md.lower().find(kw)
                start   = max(0, idx - 80)
                end     = min(len(page_md), idx + 120)
                snippet = page_md[start:end].replace("\n", " ").strip()
                matches.append(f"Page {page_no}: ...{snippet}...")
        if not matches:
            return f"Keyword '{keyword}' not found in any page."
        return "\n".join(matches)

    return [get_page_count, get_document_metadata, get_page_markdown,
            get_full_markdown, get_tables, search_pages_for_keyword]


# ------------------------------------------------------------------
# Cached document parsing
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def parse_document_from_url(url: str):
    """Parse document from URL. Result is cached per unique URL."""
    return DocumentConverter().convert(url).document


@st.cache_resource(show_spinner=False)
def parse_document_from_bytes(file_bytes: bytes, filename: str):
    """Parse an uploaded file from raw bytes. Cached by content."""
    import tempfile, pathlib
    suffix = pathlib.Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    return DocumentConverter().convert(tmp_path).document


@st.cache_resource(show_spinner=False)
def build_agent(_document):
    """Build and cache the AgentExecutor for a given document."""
    llm = ChatNVIDIA(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        temperature=0.3,
        api_key=NVIDIA_API_KEY,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    tools = build_tools(_document)
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Docling Document Agent",
        page_icon="📄",
        layout="wide",
    )

    # ---- Sidebar -------------------------------------------------------
    with st.sidebar:
        st.title("📄 Docling Agent")
        st.caption("Docling + NVIDIA NIM + LangGraph")
        st.divider()

        st.subheader("Load a Document")
        input_mode = st.radio("Input type", ["URL", "Upload file"], horizontal=True)

        pending_source = None

        if input_mode == "URL":
            url_input = st.text_input("Document URL", value=DEFAULT_SOURCE)
            if st.button("🔄 Load Document", use_container_width=True, type="primary"):
                if url_input.strip():
                    pending_source = ("url", url_input.strip())
        else:
            uploaded = st.file_uploader(
                "Upload document",
                type=["pdf", "docx", "pptx", "html", "md", "txt", "png", "jpg"],
            )
            if uploaded:
                pending_source = ("file", uploaded.read(), uploaded.name)

        st.divider()

        if "doc_info" in st.session_state:
            info = st.session_state["doc_info"]
            st.success(f"**Loaded:** {info['name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Pages",  info["pages"])
            c2.metric("Images", info["images"])
            c3.metric("Tables", info["tables"])
            if st.button("🗑 Clear / Load new", use_container_width=True):
                for k in ["doc_info", "agent", "messages"]:
                    st.session_state.pop(k, None)
                st.cache_resource.clear()
                st.rerun()
        else:
            st.info("No document loaded yet.")

    # ---- Parse & initialize agent when new source provided -------------
    if pending_source and "agent" not in st.session_state:
        with st.spinner("📖 Parsing document… first load may take a moment."):
            try:
                if pending_source[0] == "url":
                    _, url   = pending_source
                    document = parse_document_from_url(url)
                    name     = url.split("/")[-1] or url
                else:
                    _, fbytes, fname = pending_source
                    document = parse_document_from_bytes(fbytes, fname)
                    name     = fname

                st.session_state["doc_info"] = {
                    "name":   name,
                    "pages":  document.num_pages(),
                    "images": len(document.pictures),
                    "tables": len(document.tables),
                }
                st.session_state["agent"]    = build_agent(document)
                st.session_state["messages"] = []
                st.rerun()

            except Exception as e:
                st.error(f"Failed to parse document: {e}")

    # ---- Main chat area ------------------------------------------------
    st.title("📄 Document Analysis Agent")

    if "agent" not in st.session_state:
        st.info(
            "**👈 Load a document from the sidebar** to get started.\n\n"
            "Supported: PDF, DOCX, PPTX, HTML, Markdown, PNG, JPG"
        )
        st.stop()

    # Render history
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New user message
    if user_input := st.chat_input("Ask anything about the document…"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build LangChain message history (excluding the current message)
        chat_history = []
        for m in st.session_state["messages"][:-1]:
            if m["role"] == "user":
                chat_history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                chat_history.append(AIMessage(content=m["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state["agent"].invoke({
                        "input":        user_input,
                        "chat_history": chat_history,
                    })
                    response_text = result.get("output", "")

                    # Show tool calls from intermediate_steps
                    steps = result.get("intermediate_steps", [])
                    if steps:
                        with st.expander(f"🔧 Tools used ({len(steps)})", expanded=False):
                            for action, observation in steps:
                                tool_name = getattr(action, "tool", "tool")
                                tool_input = str(getattr(action, "tool_input", ""))
                                preview = str(observation)[:600]
                                if len(str(observation)) > 600:
                                    preview += "\n…(truncated)"
                                st.markdown(f"**`{tool_name}`** ← `{tool_input}`")
                                st.code(preview, language="markdown")

                    st.markdown(response_text)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response_text}
                    )

                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": err}
                    )


if __name__ == "__main__":
    main()
