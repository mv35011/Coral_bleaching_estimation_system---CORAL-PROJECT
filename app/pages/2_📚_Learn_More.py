import streamlit as st
from pathlib import Path


img_dir = Path(__file__).resolve().parents[1] / "images"

st.set_page_config(
    page_title="Learn More - Project CORAL",
    page_icon="🐠",
    layout="wide",
)

st.title("📚 Learn More About Coral Reefs")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "What is a Coral Reef?", 
    "Why Are Reefs Important?", 
    "The Threat: Coral Bleaching", 
    "How This App Helps"
])

with tab1:
    st.header("What is a Coral Reef?")

    img_url_1 = img_dir / "coral_polyps.jpg"
    st.image(str(img_url_1), caption="Corals are made of thousands of tiny animals called polyps.")
    
    st.markdown("""
    A coral reef is a massive underwater structure made from the skeletons of marine invertebrates called coral. Each individual coral is a tiny animal called a **polyp**. 

    These polyps live in large colonies and secrete a hard carbonate exoskeleton underneath them. Over thousands of years, these colonies grow and merge to form the vast, complex structures we know as reefs.

    Most reef-building corals have a vital, symbiotic relationship with a microscopic algae called **zooxanthellae** (zoh-zan-THEL-ee). The algae live inside the coral's tissue, providing the coral with up to 90% of its food through photosynthesis. In return, the coral gives the algae a protected home and the compounds it needs for photosynthesis. This algae is also what gives corals their vibrant colors.
    """)

with tab2:
    st.header("Why Are Reefs So Important?")
    
    img_url_2 = img_dir / "vibrant_reef.jpg"
    st.image(str(img_url_2), caption="Often called 'Rainforests of the Sea', reefs are biodiversity hotspots.")

    st.markdown("""
    Coral reefs are one of the most vital and diverse ecosystems on Earth. Despite covering less than 1% of the ocean floor, they support an astonishing **25% of all marine life**.

    Their importance can be broken down into three main categories:

    * **1. Biodiversity:** Reefs provide food and shelter for hundreds of thousands of species, from tiny crustaceans to large predators like sharks and groupers.
        
    * **2. Coastal Protection:** Their massive, complex structures act as a natural, living barrier that breaks up large ocean waves. This protects coastal communities, shorelines, and critical habitats like mangroves from storm surges and erosion.
        
    * **3. Economic Value:** Healthy reefs are the foundation of a massive global economy. They support commercial and subsistence fishing, provide jobs, and drive a multi-billion dollar tourism industry.
    """)

with tab3:
    st.header("The Threat: Coral Bleaching")
    
    img_url_3 = img_dir / "bleached_reef.jpg"
    st.image(str(img_url_3), caption="A bleached reef has expelled its colorful algae, leaving it a stark white.")

    st.markdown("""
    **Coral bleaching is a stress response, not instant death.** When ocean water gets too warm (even by just 1–2°C) for too long, the symbiotic relationship between the coral and its algae breaks down.

    The coral becomes stressed and **expels the colorful algae** from its tissues. 

    Without the algae, the coral's bright white skeleton becomes visible through its transparent tissue, making it look "bleached." A bleached coral is not dead, but it is **starving and highly vulnerable** to disease. If the water temperatures return to normal, the coral can slowly regain its algae and recover. If the heat stress continues, the coral will die.

    This is why **Degree Heating Weeks (DHW)**, which this app models, is such a critical metric. It measures *how long* the water has been *too hot*, which is the primary driver of these mass bleaching events.
    """)

with tab4:
    st.header("How This App Helps")
    
    img_url_4 = img_dir / "ai_monitoring.webp"
    st.image(str(img_url_4), caption="This app uses the same data scientists use to monitor reefs from space.")
    
    st.markdown("""
    This application, **Project CORAL**, is a proof-of-concept for an AI-powered early warning system.
    
    1.  **It Gathers Data:** It connects to live satellite data from NOAA (National Oceanic and Atmospheric Administration) to get real-time data on Sea Surface Temperature (SST) and Degree Heating Weeks (DHW).
        
    2.  **It Predicts Risk:** It feeds this live data into a machine learning model (a `GradientBoostingRegressor`) that was trained on over 30 years of historical data. The model's output is the predicted bleaching risk you see on the dashboard.
        
    3.  **It Empowers Users:** By providing a live dashboard, a "What-If" simulator, and a conversational AI (Finley), this tool makes complex scientific data accessible to everyone.
    
    Early warnings from tools like this can help scientists, conservation groups, and local managers make critical decisions, such as closing reefs to tourism to reduce stress or targeting efforts to protect the most resilient coral species.
    """)
