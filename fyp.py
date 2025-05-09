import pickle
import streamlit as st
import openrouteservice
from pathlib import Path
import folium
from web3 import Web3
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import random
import datetime
from geopy.distance import geodesic
import time

# Page configuration
st.set_page_config(
    page_title="WayBetter",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .stButton>button {
        background-color: #1e3d59;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2b587a;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-message {
        background-color: #e1f3e1;
        border-left: 5px solid #4caf50;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    div.stMarkdown {
        font-family: 'Source Sans Pro', sans-serif;
    }
    .eco-score {
        font-size: 18px;
        font-weight: 600;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: rgba(30, 61, 89, 0.1);
        display: inline-block;
    }
    .route-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .sidebar-content {
        padding: 20px;
        background-color: rgba(255,255,255,0.8);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session states for preserving data between reruns
if 'congestion_data' not in st.session_state:
    st.session_state.congestion_data = None
if 'route_scores' not in st.session_state:
    st.session_state.route_scores = None
if 'best_route_info' not in st.session_state:
    st.session_state.best_route_info = None
if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False
if 'reward_claimed' not in st.session_state:
    st.session_state.reward_claimed = False
if 'showing_wallet_input' not in st.session_state:
    st.session_state.showing_wallet_input = False
if 'loading' not in st.session_state:
    st.session_state.loading = False
if 'animation_complete' not in st.session_state:
    st.session_state.animation_complete = False

# Load AQI dataset
@st.cache_data
def load_data():
    df = pd.read_csv('pm25_only_aqi_dataset_updated.csv')
    df['From Date'] = pd.to_datetime(df['From Date'])
    return df

df = load_data()

# Load ARIMA models
@st.cache_resource
def load_models():
    with open('arima_models.pkl', 'rb') as f:
        return pickle.load(f)

arima_models = load_models()

# Web3 Blockchain connection
ganache_url = "HTTP://127.0.0.1:8545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Smart Contract details
contract_address = "API_KEY"
contract_abi = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "sender",
                "type": "address"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "amount",
                "type": "uint256"
            }
        ],
        "name": "Funded",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "user",
                "type": "address"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "amount",
                "type": "uint256"
            }
        ],
        "name": "RewardGiven",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "user",
                "type": "address"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "amount",
                "type": "uint256"
            }
        ],
        "name": "RewardWithdrawn",
        "type": "event"
    },
    {
        "inputs": [
            {
                "internalType": "address",
                "name": "user",
                "type": "address"
            }
        ],
        "name": "checkReward",
        "outputs": [
            {
                "internalType": "uint256",
                "name": "",
                "type": "uint256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "fundContract",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getBalance",
        "outputs": [
            {
                "internalType": "uint256",
                "name": "",
                "type": "uint256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "address",
                "name": "user",
                "type": "address"
            }
        ],
        "name": "giveReward",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [
            {
                "internalType": "address",
                "name": "",
                "type": "address"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "address",
                "name": "",
                "type": "address"
            }
        ],
        "name": "rewards",
        "outputs": [
            {
                "internalType": "uint256",
                "name": "",
                "type": "uint256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "withdrawReward",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "stateMutability": "payable",
        "type": "receive"
    }
]

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Sidebar with contract info and app description
with st.sidebar:
    #st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    
    # App logo and title
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("# 🌿")
    with col2:
        st.markdown("# WayBetter")
    
    st.markdown("---")
    
    # App description
    st.markdown("""
    **WayBetter** helps you find eco-friendly travel paths in Mumbai based on:
    
    * 🌫️ Air Quality Index
    * 🚦 Traffic Congestion 
    * ⏱️ Travel Time
    
    Choose the greenest route and earn crypto rewards for your eco-conscious choices!
    """)
    
    st.markdown("---")
    
    # Blockchain connection status
    if web3.is_connected():
        st.success("✅ Blockchain Connected")
    else:
        st.error("⚠️ Blockchain Disconnected")
        st.stop()
    
    # Contract information
    try:
        contract_owner = contract.functions.owner().call()
        contract_balance = web3.from_wei(contract.functions.getBalance().call(), 'ether')
        
        st.markdown("### Smart Contract")
        st.markdown(f"**Balance:** {contract_balance:.4f} ETH")
        
        # Visual progress bar for contract balance
        st.progress(min(float(contract_balance), 10)/10)
        
        if contract_balance < 1:
            st.warning("⚠️ Low contract balance for rewards")
    except Exception as e:
        st.error(f"Error fetching contract data")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Mumbai Locations
places = {
    'Bandra Kurla Complex': (19.0680, 72.8774),
    'Bandra': (19.0600, 72.8355),
    'Borivali East': (19.2306, 72.8598),
    'Chakala Andheri East': (19.1156, 72.8570),
    'Colaba': (18.9067, 72.8147),
    'Deonar': (19.0474, 72.9180),
    'Kandivali East': (19.2068, 72.8747),
    'Khindipada Bhandup West': (19.1549, 72.9366),
    'Kurla': (19.0726, 72.8820),
    'Malad West': (19.1862, 72.8484),
    'Mazgaon': (18.9636, 72.8411),
    'Mulund West': (19.1726, 72.9421),
    'Navy Nagar Colaba': (18.8922, 72.8122),
    'Powai': (19.1177, 72.9106),
    'Siddharth Nagar Worli': (19.0030, 72.8150),
    'Sion': (19.0421, 72.8612),
    'Vasai West': (19.3730, 72.8324),
    'Vile Parle West': (19.0991, 72.8363),
    'Worli': (18.9949, 72.8152)
}

# AQI prediction
def predict_aqi(location, date_time):
    df_loc = df[df['Location'] == location]
    closest_time = df_loc.iloc[(df_loc['From Date'] - date_time).abs().argsort()[:1]]
    
    if not closest_time.empty:
        return round(closest_time['PM2.5'].values[0])

    if location in arima_models:
        model = arima_models[location]
        time_diff = (date_time - df_loc['From Date'].max()).total_seconds() / 900
        forecast = model.forecast(steps=max(1, int(time_diff)))
        return round(forecast.iloc[-1]) if isinstance(forecast, pd.Series) else round(forecast[-1])
    
    return random.randint(50, 300)

# Find nearest known place
def find_nearest_place(lat, lon, places, threshold_km=1.0):
    closest_place = None
    min_distance = float('inf')
    for place_name, (p_lat, p_lon) in places.items():
        dist = geodesic((lat, lon), (p_lat, p_lon)).km
        if dist < threshold_km and dist < min_distance:
            min_distance = dist
            closest_place = place_name
    return closest_place

def show_wallet_input():
    st.session_state.showing_wallet_input = True

# Function to set loading state
def set_loading():
    st.session_state.loading = True
    return True

# Header animation
def header_animation():
    header_placeholder = st.empty()
    
    for i in range(5):
        if i == 0:
            header_placeholder.markdown("<h1 style='text-align: center; opacity: 0.2;'>🌿 WayBetter</h1>", unsafe_allow_html=True)
        elif i == 1:
            header_placeholder.markdown("<h1 style='text-align: center; opacity: 0.4;'>🌿 WayBetter</h1>", unsafe_allow_html=True)
        elif i == 2:
            header_placeholder.markdown("<h1 style='text-align: center; opacity: 0.6;'>🌿 WayBetter</h1>", unsafe_allow_html=True)
        elif i == 3:
            header_placeholder.markdown("<h1 style='text-align: center; opacity: 0.8;'>🌿 WayBetter</h1>", unsafe_allow_html=True)
        elif i == 4:
            header_placeholder.markdown("<h1 style='text-align: center; opacity: 1.0;'>🌿 WayBetter</h1>", unsafe_allow_html=True)
        time.sleep(0.1)
    
    st.session_state.animation_complete = True
    
    header_placeholder.markdown("""
    <h1 style='text-align: center;'>🌿 WayBetter</h1>
    <p style='text-align: center; font-size: 1.2em; margin-bottom: 40px;'>Find the most eco-friendly route through Mumbai</p>
    """, unsafe_allow_html=True)

# Run the header animation once
if not st.session_state.animation_complete:
    header_animation()
else:
    st.markdown("""
    <h1 style='text-align: center;'>🌿 WayBetter</h1>
    <p style='text-align: center; font-size: 1.2em; margin-bottom: 40px;'>Find the most eco-friendly route through Mumbai</p>
    """, unsafe_allow_html=True)

# Main content container
main_container = st.container()

with main_container:
    # Form for input parameters to control execution
    with st.form(key="route_parameters_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚏 Origin")
            source = st.selectbox("Select Source", list(places.keys()), index=0)
        
        with col2:
            st.markdown("### 🏁 Destination")
            destination = st.selectbox("Select Destination", list(places.keys()), index=10)
        
        st.markdown("### 📅 When are you traveling?")
        col3, col4 = st.columns(2)
        
        with col3:
            user_date = st.date_input("Select Date", value=datetime.date.today())
        
        with col4:
            user_time = st.time_input("Select Time", value=datetime.time(9, 0))
        
        calculate_button = st.form_submit_button("🔍 Find Best Routes", on_click=set_loading)

        if calculate_button:
            st.session_state.calculation_done = True
            # Reset reward claim state when recalculating
            st.session_state.reward_claimed = False
            st.session_state.showing_wallet_input = False
            st.session_state.congestion_data = None
            st.session_state.route_scores = None
            st.session_state.best_route_info = None

    # Show loading animation
    if st.session_state.loading and not st.session_state.route_scores:
        with st.spinner("🔍 Analyzing routes and calculating eco-scores..."):
            # Simulate some processing time for visual effect
            time.sleep(1.5)
            st.session_state.loading = False

    # Display results
    if st.session_state.calculation_done:
        user_datetime = datetime.datetime.combine(user_date, user_time)
        st.markdown(f"<p style='text-align: center;'>Planning your journey for <b>{user_datetime.strftime('%A, %B %d at %I:%M %p')}</b></p>", unsafe_allow_html=True)

        # Generate congestion data only once for this calculation
        if st.session_state.congestion_data is None:
            # Set a fixed seed for reproducibility within a session
            random.seed(42)
            st.session_state.congestion_data = {place: random.randint(10, 100) for place in places}
        
        # Only calculate routes if not already calculated
        if st.session_state.route_scores is None:
            # OpenRouteService
            API_KEY = "5b3ce3597851110001cf6248531a9782480a43888f5aa818a94ccc4a"
            client = openrouteservice.Client(key=API_KEY)

            preferences = ["recommended", "fastest", "shortest"]
            routes = []

            for pref in preferences:
                try:
                    route = client.directions(
                        coordinates=[places[source][::-1], places[destination][::-1]],
                        profile='driving-car',
                        format='geojson',
                        preference=pref
                    )
                    routes.append((route, pref))
                except Exception as e:
                    st.error(f"Error fetching {pref} route: {e}")

            # Calculate eco-scores
            route_scores = []
            for route, pref in routes:
                coordinates = route['features'][0]['geometry']['coordinates']
                
                matched_places = []
                for lon, lat in coordinates:
                    place = find_nearest_place(lat, lon, places)
                    if place and (not matched_places or matched_places[-1] != place):
                        matched_places.append(place)
                
                if not matched_places:
                    continue

                aqis = [predict_aqi(place, user_datetime) for place in matched_places]
                congestions = [st.session_state.congestion_data[place] for place in matched_places]

                avg_aqi = sum(aqis) / len(aqis)
                avg_congestion = sum(congestions) / len(congestions)

                # Calculate distance and duration
                distance = route['features'][0]['properties']['summary']['distance'] / 1000  # km
                duration = route['features'][0]['properties']['summary']['duration'] / 60  # minutes

                eco_score = avg_aqi + avg_congestion
                route_scores.append((route, eco_score, pref, matched_places, distance, duration, avg_aqi, avg_congestion))
            
            # Store in session state
            st.session_state.route_scores = route_scores

        if not st.session_state.route_scores:
            st.error("No valid routes found with matched points.")
        else:
            # Sort by eco-score and get the best route
            sorted_routes = sorted(st.session_state.route_scores, key=lambda x: x[1])
            best_route, best_score, best_pref, best_places, best_distance, best_duration, best_aqi, best_congestion = sorted_routes[0]
            
            # Store best route info for reward claiming
            if st.session_state.best_route_info is None:
                st.session_state.best_route_info = (best_route, best_score, best_pref, best_places, best_distance, best_duration)
            
            # Map
            st.markdown("### 🗺️ Route Map")
            
            # Create a map centered on Mumbai
            mumbai_map = folium.Map(location=[19.0760, 72.8777], zoom_start=11, tiles="OpenStreetMap")
            
            # Add all routes to the map with different colors
            colors = ['blue', 'purple', 'orange']
            for idx, (route, _, pref, _, _, _, _, _) in enumerate(sorted_routes):
                if idx == 0:  # Best route
                    folium.GeoJson(
                        route, 
                        name=f'Eco-Friendly Route ({pref.capitalize()})',
                        style_function=lambda _: {'color': 'green', 'weight': 5, 'opacity': 0.8}
                    ).add_to(mumbai_map)
                else:
                    folium.GeoJson(
                        route, 
                        name=f'{pref.capitalize()} Route',
                        style_function=lambda _: {'color': colors[idx-1], 'weight': 3, 'opacity': 0.6}
                    ).add_to(mumbai_map)
            
            # Add markers for source and destination
            folium.Marker(
                places[source], 
                popup=f"Start: {source}", 
                tooltip=f"Start: {source}",
                icon=folium.Icon(color="green", icon="play", prefix="fa")
            ).add_to(mumbai_map)
            
            folium.Marker(
                places[destination], 
                popup=f"End: {destination}", 
                tooltip=f"End: {destination}",
                icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
            ).add_to(mumbai_map)
            
            # Add layer control
            folium.LayerControl().add_to(mumbai_map)
            
            # Display the map
            folium_static(mumbai_map)
            
            # Results section
            st.markdown("<h3 style='text-align: center;'>📊 Route Analysis</h3>", unsafe_allow_html=True)
            
            # Best Route Card
            st.markdown("""
            <div class="success-message">
                <h3 style="margin-top:0; color: black;">🌱 Recommended Eco-Friendly Route</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create 3 columns for the best route details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Route Type")
                st.markdown(f"**{best_pref.capitalize()}**")
                st.markdown(f"**Distance:** {best_distance:.1f} km")
                st.markdown(f"**Duration:** {best_duration:.0f} min")
            
            with col2:
                st.markdown("#### Environmental Impact")
                st.markdown(f"**Air Quality:** {best_aqi:.1f}")
                st.markdown(f"**Congestion:** {best_congestion:.1f}")
                st.markdown(f"**Eco-Score:** {best_score:.1f}")
            
            with col3:
                st.markdown("#### Route Path")
                st.markdown(f"**{source}** → ... → **{destination}**")
                total_places = len(best_places)
                if total_places <= 4:
                    st.markdown(" → ".join(best_places))
                else:
                    st.markdown(f"{best_places[0]} → {best_places[1]} → ... → {best_places[-2]} → {best_places[-1]}")
            
            # Comparison table
            st.markdown("### 🔄 All Routes Comparison")
            
            # Create cards for each route
            for i, (route, score, pref, matched_places, distance, duration, avg_aqi, avg_congestion) in enumerate(sorted_routes):
                color = "#3a7ca5" if i == 0 else "#6c757d"  # Changed from "green"/"gray" to blue/dark gray
                badge = "🥇 BEST CHOICE" if i == 0 else ""
                            
                st.markdown(f"""
                <div class="route-card" style="border-left: 4px solid {color}; background-color: #ffffff;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #2c3e50;">Route {i+1}: {pref.capitalize()}</h4>
                        <span style="color: #3a7ca5; font-weight: bold;">{badge}</span>
                    </div>
                    <div style="display: flex; margin-top: 10px;">
                        <div style="flex: 1;">
                            <p style="color: #333333;"><b>Distance:</b> {distance:.1f} km</p>
                            <p style="color: #333333;"><b>Duration:</b> {duration:.0f} min</p>
                        </div>
                        <div style="flex: 1;">
                            <p style="color: #333333;"><b>Air Quality:</b> {avg_aqi:.1f}</p>
                            <p style="color: #333333;"><b>Congestion:</b> {avg_congestion:.1f}</p>
                        </div>
                        <div style="flex: 1;">
                            <p style="color: #333333;"><b>Eco-Score:</b> <span class="eco-score" style="background-color: #e8f4f8; color: #2c3e50;">{score:.1f}</span></p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            # Reward system box
            st.markdown("""
            <div style="background-color: #e8f4f8; border-radius: 10px; padding: 20px; margin-top: 30px; border-left: 5px solid #3a7ca5;">
                <h3 style="margin-top: 0; color: #2c3e50;">💰 Reward for Eco-Friendly Choices</h3>
                <p style="color: #333333;">Choose the eco-friendly route and earn cryptocurrency rewards!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Only show the claim button if not already claimed
            if not st.session_state.reward_claimed:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚗 I took the eco-friendly route!", 
                               help="Click to claim your reward for choosing the eco-friendly route",
                               on_click=show_wallet_input):
                        st.session_state.reward_claimed = True
            
            # Show wallet input if button was clicked
            if st.session_state.showing_wallet_input:
                #st.markdown("<div style='background-color: #f6f6f6; padding: 20px; border-radius: 10px; margin-top: 20px;'>", unsafe_allow_html=True)
                user_wallet = st.text_input("Enter your Ethereum wallet address", 
                                          key="wallet_address", 
                                          placeholder="0x...")
                
                if user_wallet:
                    if not web3.is_address(user_wallet):
                        st.error("⚠️ Invalid Ethereum address. Please enter a valid address.")
                    else:
                        # Check contract balance first
                        contract_balance = contract.functions.getBalance().call()
                        if contract_balance < web3.to_wei(1, 'ether'):
                            st.error("⚠️ Contract doesn't have enough ETH for rewards. Please fund it first.")
                        else:
                            # Select an account that is the owner
                            try:
                                # Get owner address from contract
                                owner_address = contract.functions.owner().call()
                                
                                # Check available accounts
                                accounts = web3.eth.accounts
                                
                                if owner_address not in accounts:
                                    st.error(f"⚠️ Contract owner account ({owner_address}) not found in available accounts. Please use the correct account that owns the contract.")
                                else:
                                    # Use the owner account to send the transaction
                                    with st.spinner("Processing transaction..."):
                                        # Estimate gas for the transaction (to avoid out-of-gas errors)
                                        gas_estimate = contract.functions.giveReward(user_wallet).estimate_gas({'from': owner_address})
                                        
                                        # Execute the transaction with proper gas
                                        tx = contract.functions.giveReward(user_wallet).transact({
                                            'from': owner_address,
                                            'gas': int(gas_estimate * 1.2)  # Add 20% buffer
                                        })
                                        
                                        # Wait for transaction receipt
                                        receipt = web3.eth.wait_for_transaction_receipt(tx)
                                        
                                        if receipt.status == 1:
                                            success_animation = st.empty()
                                            for i in range(5):
                                                if i % 2 == 0:
                                                    success_animation.markdown("""
                                                    <div style="text-align: center; padding: 20px;">
                                                        <h2 style="color: #4CAF50; transform: scale(1.1); transition: all 0.3s ease;">🎉 REWARD SENT! 🎉</h2>
                                                        <div style="font-size: 3rem; margin: 10px 0;">💰</div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                else:
                                                    success_animation.markdown("""
                                                    <div style="text-align: center; padding: 20px;">
                                                        <h2 style="color: #3a7ca5; transform: scale(1.0); transition: all 0.3s ease;">🎉 REWARD SENT! 🎉</h2>
                                                        <div style="font-size: 3rem; margin: 10px 0;">✨</div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                time.sleep(0.3)
                                            
                                            # Final success message with animation
                                            success_animation.markdown("""
                                            <div style="text-align: center; padding: 20px; animation: fadeInUp 1s ease;">
                                                <h2 style="color: #4CAF50;">🎉 REWARD SENT SUCCESSFULLY! 🎉</h2>
                                                <div style="font-size: 3rem; margin: 15px 0;">💰✨</div>
                                                <style>
                                                @keyframes fadeInUp {
                                                    from {
                                                        opacity: 0;
                                                        transform: translate3d(0, 40px, 0);
                                                    }
                                                    to {
                                                        opacity: 1;
                                                        transform: translate3d(0, 0, 0);
                                                    }
                                                }
                                                </style>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            st.balloons()  # Keep the balloons effect
                                            
                                            # Add confetti animation
                                            st.markdown("""
                                            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; z-index: 1000;">
                                                <div id="confetti-container"></div>
                                            </div>
                                            <script>
                                            const confettiColors = ['#3a7ca5', '#4CAF50', '#FFC107', '#FF5722'];
                                            const confettiCount = 150;
                                            const confettiContainer = document.getElementById('confetti-container');
                                            
                                            for (let i = 0; i < confettiCount; i++) {
                                                const confetti = document.createElement('div');
                                                confetti.style.position = 'absolute';
                                                confetti.style.width = Math.random() * 10 + 5 + 'px';
                                                confetti.style.height = Math.random() * 5 + 5 + 'px';
                                                confetti.style.backgroundColor = confettiColors[Math.floor(Math.random() * confettiColors.length)];
                                                confetti.style.left = Math.random() * 100 + 'vw';
                                                confetti.style.top = -20 + 'px';
                                                confetti.style.borderRadius = '50%';
                                                confetti.style.opacity = Math.random() * 0.8 + 0.2;
                                                confetti.style.animation = 'fall ' + (Math.random() * 3 + 2) + 's linear forwards';
                                                confettiContainer.appendChild(confetti);
                                            }
                                            
                                            
                                            </script>
                                            """, unsafe_allow_html=True)
                                            
                                            st.success("✅ Reward successfully sent!")
                                            st.markdown(f"**Transaction Hash:** `{web3.to_hex(tx)}`")
                                            
                                            # Show updated reward balance with animation
                                            new_reward = web3.from_wei(contract.functions.checkReward(user_wallet).call(), 'ether')
                                            st.markdown(f"""
                                            <div style="background-color: #e8f4f8; border-radius: 10px; padding: 15px; animation: pulse 2s infinite;">
                                                <h3 style="margin: 0; color: #2c3e50;">Your reward balance:</h3>
                                                <p style="font-size: 1.5rem; font-weight: bold; color: #3a7ca5; margin: 5px 0;">
                                                    {new_reward} ETH
                                                </p>
                                                <style>
                                                @keyframes pulse {{
                                                    0% {{ box-shadow: 0 0 0 0 rgba(58, 124, 165, 0.4); }}
                                                    70% {{ box-shadow: 0 0 0 10px rgba(58, 124, 165, 0); }}
                                                    100% {{ box-shadow: 0 0 0 0 rgba(58, 124, 165, 0); }}
                                                }}
                                                </style>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            st.error("⚠️ Transaction failed. See details in the Ethereum node logs.")
                            except Exception as e:
                                st.error(f"⚠️ Error during transaction: {str(e)}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a section for contract funding (optional)
            with st.expander("💼 Admin: Fund the Contract"):
                st.write("Only use this if you are the contract owner and want to add funds.")
                fund_amount = st.number_input("Amount to fund (ETH)", min_value=0.1, step=0.1)
                fund_account = st.selectbox("Select account to fund from", web3.eth.accounts)
                
                if st.button("💲 Fund Contract"):
                    with st.spinner("Processing funding transaction..."):
                        try:
                            tx = contract.functions.fundContract().transact({
                                'from': fund_account,
                                'value': web3.to_wei(fund_amount, 'ether')
                            })
                            web3.eth.wait_for_transaction_receipt(tx)
                            st.success(f"✅ Contract funded with {fund_amount} ETH!")
                            
                            # Update contract balance display
                            # Update contract balance display
                            new_balance = web3.from_wei(contract.functions.getBalance().call(), 'ether')
                            st.write(f"New contract balance: {new_balance} ETH")
                        except Exception as e:
                            st.error(f"⚠️ Error funding contract: {str(e)}")
    
# Reset button to clear all calculations and start fresh
if st.session_state.calculation_done:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Reset and Calculate New Route", help="Clear current calculations and start over"):
            st.session_state.congestion_data = None
            st.session_state.route_scores = None
            st.session_state.best_route_info = None
            st.session_state.calculation_done = False
            st.session_state.reward_claimed = False
            st.session_state.showing_wallet_input = False
            st.rerun()

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666;">
    <p>WayBetter: Making transportation in Mumbai sustainable, one route at a time.</p>
    <p>© 2025 WayBetter | Built by AFJAN</p>
</div>
""", unsafe_allow_html=True)
