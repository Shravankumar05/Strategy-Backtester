import streamlit as st

def section_header(title, level=2, description=None, divider=True):
    styles = {
        1: {
            'font-size': '1.75rem',
            'font-weight': '700',
            'margin': '1.5rem 0 0.5rem',
            'color': '#1e3c72',
            'border-bottom': '2px solid #e9ecef',
            'padding-bottom': '0.5rem'
        },
        2: {
            'font-size': '1.5rem',
            'font-weight': '600',
            'margin': '1.25rem 0 0.5rem',
            'color': '#2a5298',
            'border-left': '4px solid #2a5298',
            'padding-left': '0.75rem'
        },
        3: {
            'font-size': '1.25rem',
            'font-weight': '600',
            'margin': '1rem 0 0.5rem',
            'color': '#3a6fb0',
            'border-left': '3px solid #3a6fb0',
            'padding-left': '0.75rem'
        },
        4: {
            'font-size': '1.1rem',
            'font-weight': '600',
            'margin': '0.75rem 0 0.5rem',
            'color': '#4a7cb8',
            'font-style': 'italic'
        },
        5: {
            'font-size': '1rem',
            'font-weight': '500',
            'margin': '0.5rem 0 0.25rem',
            'color': '#5a8cc0',
            'text-decoration': 'underline'
        },
        6: {
            'font-size': '0.9rem',
            'font-weight': '500',
            'margin': '0.5rem 0 0.25rem',
            'color': '#6a9cc8',
            'font-style': 'italic'
        }
    }
    
    if level in styles:
        style = '; '.join([f"{k}: {v}" for k, v in styles[level].items()])
    else:
        # Fallback to level 4 style for any unknown levels
        style = '; '.join([f"{k}: {v}" for k, v in styles[4].items()])
    
    st.markdown(f'<div style="{style}">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div style="color: #4a5568; margin-bottom: 0.5rem;">{description}</div>', 
                   unsafe_allow_html=True)
    
    if divider and level < 3:  # Only add divider for h1 and h2
        st.markdown("<hr style='margin: 0.5rem 0 1rem; border-top: 1px solid #e9ecef;'/>", 
                   unsafe_allow_html=True)

def card(title, content, level=3, background='#f8f9fa', padding='1rem', margin='0 0 1rem'):
    st.markdown(
        f"""
        <div style="
            background: {background};
            border-radius: 8px;
            padding: {padding};
            margin: {margin};
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            {f'<h{level} style="margin-top: 0; color: #2a5298;">{title}</h{level}>' if title else ''}
            {content if isinstance(content, str) else ''}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if callable(content):
        content()

def apply_global_styles():
    st.markdown(
        """
        <style>
            /* Global color scheme - Clean Blue on Gray/White theme */
            :root {
                --primary-blue: #1e3c72;
                --secondary-blue: #2a5298;
                --light-gray: #f8f9fa;
                --medium-gray: #e9ecef;
                --white: #ffffff;
                --text-primary: #1e3c72;
                --text-secondary: #2a5298;
                --border-color: #dee2e6;
            }
            
            /* Remove Streamlit default styling */
            .stApp > header {
                background-color: transparent;
            }
            
            .stApp {
                background-color: #f8f9fa;
                max-width: 100%;
                margin: 0;
                padding: 0;
            }
            
            /* Main container styling */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 100%;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            
            /* Headers and text styling */
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-primary) !important;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            /* Input fields styling - Clean borders */
            .stTextInput>div>div>input,
            .stNumberInput>div>div>input,
            .stSelectbox>div>div>div,
            .stTextArea>div>textarea,
            .stDateInput>div>div>input {
                border: 1px solid var(--border-color) !important;
                border-radius: 6px !important;
                padding: 0.75rem 1rem !important;
                background: white !important;
                transition: border-color 0.2s ease !important;
            }
            
            .stTextInput>div>div>input:focus,
            .stNumberInput>div>div>input:focus,
            .stSelectbox>div>div>div:focus,
            .stTextArea>div>textarea:focus,
            .stDateInput>div>div>input:focus {
                border-color: var(--primary-blue) !important;
                box-shadow: 0 0 0 2px rgba(30, 60, 114, 0.2) !important;
            }
            
            /* Buttons styling - Clean solid colors */
            .stButton>button {
                background-color: var(--primary-blue);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                transition: background-color 0.2s ease;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .stButton>button:hover {
                background-color: var(--secondary-blue);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            }
            
            /* Dataframe styling */
            .stDataFrame {
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            }
            
            /* Metric cards styling */
            .stMetric {
                background: linear-gradient(135deg, var(--cream) 0%, white 100%);
                border-radius: 12px;
                padding: 1.5rem;
                border-left: 4px solid var(--primary-blue);
                margin: 0.5rem 0;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
                transition: all 0.3s ease;
            }
            
            .stMetric:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
            }
            
            .stMetric > div:first-child {
                color: var(--text-secondary);
                font-size: 0.9rem;
                font-weight: 600;
            }
            
            .stMetric > div:last-child {
                color: var(--text-primary);
                font-size: 1.8rem;
                font-weight: 700;
            }
            
            /* Expander styling */
            .streamlit-expanderHeader {
                background: linear-gradient(135deg, var(--cream) 0%, var(--light-blue) 100%);
                border-radius: 8px;
                border: 1px solid var(--dark-cream);
                color: var(--text-primary);
                font-weight: 600;
            }
            
            /* Progress bar styling */
            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
                border-radius: 6px;
                height: 10px;
            }
            
            /* Selectbox styling */
            .stSelectbox > div > div {
                background: white;
                border: 2px solid var(--dark-cream);
                border-radius: 8px;
            }
            
            /* Alert styling */
            .stAlert {
                border-radius: 8px;
                border-left: 4px solid var(--primary-blue);
            }
            
            /* Tables */
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 1rem 0;
                font-size: 0.9rem;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
            }
            
            th, td {
                padding: 0.75rem;
                text-align: left;
                border-bottom: 1px solid var(--dark-cream);
            }
            
            th {
                background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
                color: white;
                font-weight: 600;
            }
            
            tr:hover {
                background-color: var(--light-blue);
            }
        </style>
        """,
        unsafe_allow_html=True
    )
