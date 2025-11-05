"""
Streamlit Chat Interface for Lucky RAG System
Provides an interactive chat interface for querying technical documentation
with multimodal RAG (text + images) powered by CLIP embeddings.
"""

import streamlit as st
from PIL import Image
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional, Dict, List, Any
import traceback

# Import your RAG system
# Adjust the import based on where LuckyRAG.py is located
try:
    from LuckyRAG import (
        rag_chain,
        PROJECT_ROOT,
        IMG_DIR,
        COLLECTION_NAME,
        vectorstore
    )
except ImportError:
    st.error("""
    ⚠️ Could not import LuckyRAG module. 
    
    Please ensure:
    1. LuckyRAG.py is in the same directory as this file, OR
    2. LuckyRAG.py is in your Python path
    
    Current directory: {}
    """.format(Path.cwd()))
    st.stop()


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="HP300 Technical Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main chat container */
    .stChatMessage {
        background-color: #000000;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f1f8e9;
    }
    
    /* Diagram metadata caption */
    .diagram-metadata {
        font-size: 0.85em;
        color: #666;
        padding: 0.5rem;
        background-color: #000000;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    
    /* Source reference styling */
    .source-reference {
        background-color: #000000;
        border-left: 4px solid #ffc107;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    /* Stats display */
    .stat-box {
        background-color: #e8f4f8;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Image caption styling */
    .image-caption {
        text-align: center;
        font-size: 0.9em;
        color: #555;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    
    if "total_diagrams_shown" not in st.session_state:
        st.session_state.total_diagrams_shown = 0
    
    if "conversation_start_time" not in st.session_state:
        st.session_state.conversation_start_time = datetime.now()
    
    if "rag_errors" not in st.session_state:
        st.session_state.rag_errors = []


# ============================================================================
# CORE RAG QUERY FUNCTION
# ============================================================================

def query_rag_system(user_question: str) -> Optional[Dict[str, Any]]:
    """
    Execute the RAG chain and return results.
    
    Args:
        user_question: User's query string
        
    Returns:
        Dictionary containing answer, diagrams, and images, or None on error
    """
    try:
        # Invoke the RAG chain
        result = rag_chain.invoke({"question": user_question})
        
        # Validate result structure
        if not isinstance(result, dict):
            raise ValueError(f"RAG chain returned invalid type: {type(result)}")
        
        # Check if answer exists and has content
        if "answer" not in result:
            raise ValueError("RAG chain did not return an 'answer' field")
        
        # Log the result structure for debugging
        if not result.get("answer") or not result["answer"].strip():
            st.warning("⚠️ Empty response received from LLM. Check the retrieval results.")
            # Log retrieval info
            diagrams = result.get("diagrams", [])
            st.info(f"Retrieved {len(diagrams)} diagrams. The context might be insufficient.")
        
        # Update statistics
        st.session_state.total_queries += 1
        if result.get("diagrams"):
            st.session_state.total_diagrams_shown += len(result["diagrams"])
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        st.session_state.rag_errors.append({
            "timestamp": datetime.now(),
            "question": user_question,
            "error": error_msg,
            "traceback": traceback.format_exc()
        })
        st.error(f"❌ {error_msg}")
        with st.expander("🔍 View Error Details"):
            st.code(traceback.format_exc())
        return None


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_diagram_with_metadata(diagram: Dict[str, Any], idx: int):
    """
    Display a single diagram with its metadata.
    
    Args:
        diagram: Dictionary containing file_path, source, page, score
        idx: Index number for display
    """
    file_path = diagram.get("file_path")
    
    if file_path and Path(file_path).exists():
        try:
            # Display the image
            img = Image.open(file_path)
            st.image(img, use_container_width=300)
            
            # Display metadata
            source = diagram.get("source", "Unknown")
            page = diagram.get("page", "—")
            score = diagram.get("score")
            
            # Format metadata caption
            caption_parts = [
                f"📄 **Source:** {source}",
                f"**Page:** {page}"
            ]
            
            if score is not None:
                caption_parts.append(f"**Relevance:** {score:.3f}")
            
            st.caption(" | ".join(caption_parts))
            
        except Exception as e:
            st.warning(f"⚠️ Could not load image {idx}: {str(e)}")
    else:
        st.warning(f"⚠️ Diagram {idx} not found: {file_path}")


def display_source_references(diagrams: List[Dict[str, Any]], max_display: int = 6):
    """
    Display source references in an expandable section.
    
    Args:
        diagrams: List of diagram metadata dictionaries
        max_display: Maximum number of references to show
    """
    if not diagrams:
        return
    
    with st.expander(f"📚 View Source References ({len(diagrams)} total)", expanded=False):
        for i, d in enumerate(diagrams[:max_display], 1):
            source = d.get("source", "Unknown")
            page = d.get("page", "—")
            score = d.get("score")
            file_path = d.get("file_path", "N/A")
            
            # Create reference string
            ref_parts = [
                f"**{i}.** {source}",
                f"Page {page}"
            ]
            
            if score is not None:
                ref_parts.append(f"Score: {score:.3f}")
            
            st.markdown(" | ".join(ref_parts))
            
            # Show file path in smaller text
            if file_path != "N/A":
                st.caption(f"📁 `{Path(file_path).name}`")
            
            if i < len(diagrams[:max_display]):
                st.divider()


def display_assistant_response(result: Dict[str, Any]):
    """
    Display the complete assistant response including answer and diagrams.
    
    Args:
        result: Dictionary containing answer, diagrams, and images from RAG chain
    """
    # Display the text answer
    answer = result.get("answer", "")
    
    # Check if answer is empty or just whitespace
    if not answer or not answer.strip():
        st.warning("⚠️ No text response generated. This might be due to missing context or information outside the scope of the manual")
    elif any(phrase in answer.lower() for phrase in no_answer_phrases):
        st.markdown(answer)
        st.markdown("""
        - The question might need more context
        - The retrieval didn't find relevant information
        - There was an issue with the LLM generation
        
        **Try:**
        - Rephrasing your question
        - Being more specific
        - Asking about a different aspect
        """)
        st.error("Try Again or FEED ME BETTER DATA: Call the mechanic I never read that book or guide but let me know if I can ingest it tho :) - p.s. I have no images to show here")
        return
    
   
    st.markdown(answer)
    # Display diagrams if present
    diagrams = result.get("diagrams", [])
    

    if diagrams:
        st.markdown("---")
        st.markdown("### 🖼️ Related Diagrams")
        
        # Determine number of columns based on number of images
        num_diagrams = min(len(diagrams), 6)
        
        if num_diagrams <= 2:
            cols_per_row = num_diagrams
        else:
            cols_per_row = 2
        
        # Display images in grid layout
        for idx in range(0, num_diagrams, cols_per_row):
            cols = st.columns(cols_per_row)
            
            for col_idx, col in enumerate(cols):
                img_idx = idx + col_idx
                
                if img_idx < num_diagrams:
                    with col:
                        display_diagram_with_metadata(diagrams[img_idx], img_idx + 1)
        
        # Show source references
        display_source_references(diagrams)


# ============================================================================
# SIDEBAR COMPONENTS
# ============================================================================

def render_sidebar():
    """Render the sidebar with system info and controls"""
    with st.sidebar:
        st.markdown("# ⚙️ System Info")
        
        # System statistics
        st.markdown("### 📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", st.session_state.total_queries)
        with col2:
            st.metric("Diagrams Shown", st.session_state.total_diagrams_shown)
        
        # Session duration
        duration = datetime.now() - st.session_state.conversation_start_time
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        st.caption(f"⏱️ Session duration: {duration_str}")
        
        st.divider()
        
        # System details
        st.markdown("### 🔧 Configuration")
        
        with st.expander("View System Details", expanded=False):
            st.write(f"**Collection:** `{COLLECTION_NAME}`")
            st.write(f"**Project Root:** `{PROJECT_ROOT}`")
            st.write(f"**Image Directory:** `{IMG_DIR}`")
            
            # Try to get collection stats
            try:
                collection_count = vectorstore._collection.count()
                st.write(f"**Documents in DB:** {collection_count}")
            except Exception as e:
                st.write(f"**Documents in DB:** Unable to retrieve")
        
        st.divider()
        
        # Controls
        st.markdown("### 🎛️ Controls")
        
        # Debug mode toggle
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False
        
        st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
        
        if st.session_state.debug_mode:
            st.caption("Shows retrieval and generation details")
        
        st.divider()
        st.markdown("### 🎛️ Actions")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.session_state.total_diagrams_shown = 0
            st.session_state.conversation_start_time = datetime.now()
            st.rerun()
        
        if st.button("📥 Export Chat", use_container_width=True):
            if st.session_state.messages:
                export_chat_history()
            else:
                st.warning("No messages to export")
        
        # Error log viewer
        if st.session_state.rag_errors:
            st.divider()
            st.markdown("### ⚠️ Error Log")
            with st.expander(f"View Errors ({len(st.session_state.rag_errors)})", expanded=False):
                for idx, error in enumerate(reversed(st.session_state.rag_errors[-5:]), 1):
                    st.error(f"**Error {idx}:** {error['error']}")
                    st.caption(f"Time: {error['timestamp'].strftime('%H:%M:%S')}")
                    with st.expander("View Traceback"):
                        st.code(error['traceback'])
        
        st.divider()
        
        # Help section
        with st.expander("ℹ️ Help & Tips", expanded=False):
            st.markdown("""
            **How to use:**
            1. Type your question in the chat input
            2. Press Enter or click Send
            3. View the answer and related diagrams
            
            **Example questions:**
            - "What is the lockout-tagout procedure for the HP300?"
            - "How do I replace the accumulator?"
            - "Show me the lubrication schedule"
            - "What are the torque specifications for the main shaft?"
            
            **Tips:**
            - Be specific in your questions
            - Refer to diagrams by their page numbers
            - Use technical terms from the manual
            """)


def export_chat_history():
    """Export chat history to a text file"""
    if not st.session_state.messages:
        return
    
    export_text = f"# HP300 Chat History Export\n"
    export_text += f"# Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_text += f"# Total queries: {st.session_state.total_queries}\n\n"
    export_text += "=" * 80 + "\n\n"
    
    for idx, msg in enumerate(st.session_state.messages, 1):
        role = msg["role"].upper()
        content = msg["content"]
        timestamp = msg.get("timestamp", "N/A")
        
        export_text += f"[{idx}] {role} ({timestamp})\n"
        export_text += "-" * 80 + "\n"
        export_text += f"{content}\n\n"
        
        if msg["role"] == "assistant" and "diagrams" in msg:
            export_text += "REFERENCED DIAGRAMS:\n"
            for i, d in enumerate(msg["diagrams"], 1):
                export_text += f"  {i}. {d.get('source', 'Unknown')} - Page {d.get('page', '—')}\n"
            export_text += "\n"
        
        export_text += "=" * 80 + "\n\n"
    
    st.download_button(
        label="💾 Download Chat History",
        data=export_text,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Blacklist phrases that indicate "I don't know" responses
no_answer_phrases = [
    "provided context does not include",
    "please provide additional information",
    "context is insufficient",
    "i don't have information",
    "not found in the context",
    "context does not contain",
    'context provided does not',
    'context does not include specific information'
]

def render_chat_interface():
    """Render the main chat interface"""
    
    # Header
    st.title("🔧 HP300 Cone Crusher Technical Assistant")
    st.markdown("""
    Ask questions about the **Nordberg HP300 Cone Crusher Manual** and get step-by-step 
    procedures with relevant diagrams and safety information.
    """)
    
    st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display diagrams for assistant messages
            if message["role"] == "assistant" and "result" in message:
                result = message["result"]

                diagrams = result.get("diagrams", [])
                
                
                if diagrams:
                    st.markdown("---")
                    st.markdown("**📸 Related Diagrams**")
                    
                    # Display up to 3 diagrams in chat history (for brevity)
                    num_to_show = min(len(diagrams), 3)
                    cols = st.columns(num_to_show)
                    
                    for idx, (col, diagram) in enumerate(zip(cols, diagrams[:num_to_show])):
                        with col:
                            file_path = diagram.get("file_path")
                            if file_path and Path(file_path).exists():
                                try:
                                    img = Image.open(file_path)
                                    st.image(img, use_container_width=True)
                                    st.caption(f"Page {diagram.get('page', '—')}")
                                except Exception:
                                    pass
                    
                    if len(diagrams) > 3:
                        st.caption(f"_...and {len(diagrams) - 3} more diagrams_")
    
    # Chat input
    if prompt := st.chat_input("Ask about the HP300 Cone Crusher..."):
        # Add user message to chat
        timestamp = datetime.now().strftime('%H:%M:%S')
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documentation and generating response..."):
                result = query_rag_system(prompt)
                
                if result:
                    # Show debug info if enabled
                    if st.session_state.get("debug_mode", False):
                        with st.expander("🐛 Debug Information", expanded=False):
                            st.markdown("**Query Processing Details:**")
                            
                            # Show what was retrieved
                            diagrams = result.get("diagrams", [])
                            st.write(f"📊 Retrieved {len(diagrams)} diagrams")
                            
                            # Show answer length
                            answer = result.get("answer", "")
                            st.write(f"📝 Answer length: {len(answer)} characters")
                            
                            # Show first few diagrams metadata
                            if diagrams:
                                st.write("**Top Retrieved Diagrams:**")
                                for i, d in enumerate(diagrams[:3], 1):
                                    st.write(f"{i}. {d.get('source', 'Unknown')} - Page {d.get('page', '?')} (Score: {d.get('score', 'N/A')})")
                            
                            # Show raw result structure
                            with st.expander("Raw Result Structure"):
                                st.json({
                                    "has_answer": bool(answer and answer.strip()),
                                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                                    "num_diagrams": len(diagrams),
                                    "result_keys": list(result.keys())
                                })
                    
                    # Display the response
                    display_assistant_response(result)
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "timestamp": datetime.now().strftime('%H:%M:%S'),
                        "result": result
                    })
                else:
                    error_message = "I encountered an error processing your request. Please try rephrasing your question."
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "timestamp": datetime.now().strftime('%H:%M:%S')
                    })


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat interface
    render_chat_interface()
    
    # Footer
    st.divider()
    st.caption("""
    ⚡ Powered by OpenAI GPT-4o-mini | 🔍 CLIP Embeddings | 💾 ChromaDB  
    Built with LangChain & Streamlit
    """)


if __name__ == "__main__":
    main()