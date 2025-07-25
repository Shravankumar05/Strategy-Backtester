import streamlit as st
from typing import List, Optional, Dict, Any

class ResponsiveLayout:
    @staticmethod
    def apply_responsive_styles():
        st.markdown("""
        <style>
        /* Base responsive styles */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 100%;
        }
        
        /* Mobile-first approach */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            
            /* Stack columns on mobile */
            .stColumns > div {
                width: 100% !important;
                margin-bottom: 1rem;
            }
            
            /* Adjust metric cards for mobile */
            .metric-card {
                margin-bottom: 0.5rem;
                padding: 0.75rem;
            }
            
            /* Make charts responsive */
            .js-plotly-plot {
                width: 100% !important;
            }
            
            /* Adjust sidebar for mobile */
            .css-1d391kg {
                width: 100%;
            }
            
            /* Responsive text sizes */
            .main-header {
                font-size: 1.8rem !important;
                padding: 0.5rem !important;
            }
            
            /* Responsive tables */
            .dataframe {
                font-size: 0.8rem;
            }
            
            /* Adjust button sizes */
            .stButton > button {
                width: 100%;
                margin-bottom: 0.5rem;
            }
            
            /* Responsive tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.5rem 0.75rem;
                font-size: 0.9rem;
            }
        }
        
        /* Tablet styles */
        @media (min-width: 769px) and (max-width: 1024px) {
            .main .block-container {
                padding-left: 2rem;
                padding-right: 2rem;
            }
            
            /* Adjust column widths for tablet */
            .stColumns > div:first-child {
                width: 60% !important;
            }
            
            .stColumns > div:last-child {
                width: 40% !important;
            }
            
            /* Responsive metric cards */
            .metric-card {
                padding: 1rem;
            }
        }
        
        /* Desktop styles */
        @media (min-width: 1025px) {
            .main .block-container {
                max-width: 1200px;
                padding-left: 3rem;
                padding-right: 3rem;
            }
            
            /* Enhanced desktop layout */
            .sidebar-section {
                margin-bottom: 1.5rem;
            }
            
            /* Better spacing for desktop */
            .metric-card {
                padding: 1.25rem;
                margin-bottom: 1.5rem;
            }
        }
        
        /* Large desktop styles */
        @media (min-width: 1400px) {
            .main .block-container {
                max-width: 1400px;
            }
            
            /* Utilize extra space on large screens */
            .results-container {
                padding: 2rem;
            }
        }
        
        /* Print styles */
        @media print {
            .stSidebar {
                display: none !important;
            }
            
            .main .block-container {
                max-width: 100%;
                padding: 0;
            }
            
            .stButton {
                display: none !important;
            }
            
            /* Ensure charts print well */
            .js-plotly-plot {
                break-inside: avoid;
            }
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            .metric-card {
                border: 2px solid #000;
                background: #fff;
            }
            
            .main-header {
                border: 2px solid #000;
                background: #fff;
                color: #000;
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .metric-card {
                background: #2b2b2b;
                border-color: #444;
                color: #fff;
            }
            
            .sidebar-section {
                background: #2b2b2b;
                border-left-color: #4a9eff;
            }
            
            .results-container {
                background: #2b2b2b;
                border-color: #444;
                color: #fff;
            }
        }
        
        /* Focus styles for accessibility */
        .stButton > button:focus,
        .stSelectbox > div > div:focus,
        .stTextInput > div > div > input:focus {
            outline: 2px solid #4a9eff;
            outline-offset: 2px;
        }
        
        /* Improve readability */
        .stMarkdown {
            line-height: 1.6;
        }
        
        /* Responsive navigation */
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap;
        }
        
        /* Ensure minimum touch target size */
        .stButton > button,
        .stSelectbox > div,
        .stCheckbox > label {
            min-height: 44px;
            min-width: 44px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_responsive_columns(ratios: List[float], mobile_stack: bool = True, gap: str = "medium") -> List:
        if mobile_stack:
            st.markdown("""
            <style>
            @media (max-width: 768px) {
                .responsive-columns > div {
                    width: 100% !important;
                    margin-bottom: 1rem;
                }
            }
            </style>
            """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="responsive-columns">', unsafe_allow_html=True)
            cols = st.columns(ratios, gap=gap)
            st.markdown('</div>', unsafe_allow_html=True)
            
        return cols
    
    @staticmethod
    def create_responsive_metrics(metrics: List[Dict[str, Any]], columns_desktop: int = 4, columns_tablet: int = 2, columns_mobile: int = 1):
        st.markdown(f"""
        <style>
        .responsive-metrics {{
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat({columns_desktop}, 1fr);
        }}
        
        @media (max-width: 1024px) {{
            .responsive-metrics {{
                grid-template-columns: repeat({columns_tablet}, 1fr);
            }}
        }}
        
        @media (max-width: 768px) {{
            .responsive-metrics {{
                grid-template-columns: repeat({columns_mobile}, 1fr);
            }}
        }}
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="responsive-metrics">', unsafe_allow_html=True)
            
            cols = st.columns(columns_desktop)
            for i, metric in enumerate(metrics):
                col_index = i % columns_desktop
                with cols[col_index]:
                    st.metric(
                        label=metric.get('label', ''),
                        value=metric.get('value', ''),
                        delta=metric.get('delta', None)
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def create_responsive_chart_container(title: str, height_desktop: int = 600, height_tablet: int = 500, height_mobile: int = 400):
        st.markdown(f"""
        <style>
        .responsive-chart-container {{
            width: 100%;
            height: {height_desktop}px;
        }}
        
        @media (max-width: 1024px) {{
            .responsive-chart-container {{
                height: {height_tablet}px;
            }}
        }}
        
        @media (max-width: 768px) {{
            .responsive-chart-container {{
                height: {height_mobile}px;
            }}
        }}
        </style>
        """, unsafe_allow_html=True)
        
        st.subheader(title)
        return st.container()
    
    @staticmethod
    def create_responsive_sidebar():
        st.markdown("""
        <style>
        /* Responsive sidebar */
        @media (max-width: 768px) {
            .css-1d391kg {
                width: 100%;
                position: relative;
                transform: none;
            }
            
            .css-1d391kg .css-1v3fvcr {
                padding-top: 1rem;
            }
        }
        
        /* Collapsible sidebar on tablet */
        @media (min-width: 769px) and (max-width: 1024px) {
            .css-1d391kg {
                width: 250px;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def get_device_type() -> str:
        return "desktop" #default
    
    @staticmethod
    def create_responsive_table(df, mobile_columns: Optional[List[str]] = None, tablet_columns: Optional[List[str]] = None):
        st.markdown("""
        <style>
        .responsive-table {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        
        @media (max-width: 768px) {
            .responsive-table table {
                font-size: 0.8rem;
            }
            
            .responsive-table th,
            .responsive-table td {
                padding: 0.5rem 0.25rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="responsive-table">', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def create_mobile_friendly_tabs(tab_labels: List[str]) -> List:
        st.markdown("""
        <style>
        /* Mobile-friendly tabs */
        @media (max-width: 768px) {
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.25rem;
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.5rem 0.75rem;
                font-size: 0.85rem;
                white-space: nowrap;
                min-width: auto;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        return st.tabs(tab_labels)


def apply_global_responsive_styles():
    ResponsiveLayout.apply_responsive_styles()
    ResponsiveLayout.create_responsive_sidebar()


def create_responsive_layout_test():
    st.title("Responsive Layout Test")
    apply_global_responsive_styles()
    st.subheader("Responsive Columns")
    cols = ResponsiveLayout.create_responsive_columns([1, 1, 1])
    with cols[0]:
        st.write("Column 1 content")
        st.button("Button 1")
    with cols[1]:
        st.write("Column 2 content")
        st.button("Button 2")
    with cols[2]:
        st.write("Column 3 content")
        st.button("Button 3")
    st.subheader("Responsive Metrics")
    metrics = [
        {"label": "Metric 1", "value": "100", "delta": "10"},
        {"label": "Metric 2", "value": "200", "delta": "-5"},
        {"label": "Metric 3", "value": "300", "delta": "15"},
        {"label": "Metric 4", "value": "400", "delta": "20"},
    ]
    ResponsiveLayout.create_responsive_metrics(metrics)
    with ResponsiveLayout.create_responsive_chart_container("Test Chart"):
        st.write("Chart content would go here")
        st.line_chart({"data": [1, 2, 3, 4, 5]})
    st.subheader("Responsive Table")
    test_df = pd.DataFrame({
        "Column 1": [1, 2, 3, 4, 5],
        "Column 2": ["A", "B", "C", "D", "E"],
        "Column 3": [10.5, 20.3, 30.1, 40.8, 50.2]
    })
    ResponsiveLayout.create_responsive_table(test_df)
    st.subheader("Mobile-Friendly Tabs")
    tabs = ResponsiveLayout.create_mobile_friendly_tabs([
        "Tab 1", "Tab 2", "Tab 3", "Tab 4", "Tab 5"
    ])
    with tabs[0]:
        st.write("Content for Tab 1")
    with tabs[1]:
        st.write("Content for Tab 2")
    with tabs[2]:
        st.write("Content for Tab 3")
    with tabs[3]:
        st.write("Content for Tab 4")
    with tabs[4]:
        st.write("Content for Tab 5")