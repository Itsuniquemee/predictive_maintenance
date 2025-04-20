import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import json
import hashlib
import time
from ml_models import train_failure_prediction_model, predict_equipment_failure
from db_operations import *

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Database connection
def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Manas@7565",
            database="predictive_maintenance"
        )
        return connection
    except Error as e:
        st.error(f"Error connecting to MySQL database: {e}")
        return None

# Authentication
def check_credentials(username, password):
    conn = create_db_connection()
    if conn:
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            # In a real application, use proper password hashing
            if user['password_hash'] == hashlib.sha256(password.encode()).hexdigest():
                return user
        return None

def login_page():
    st.title("Predictive Maintenance System - Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        user = check_credentials(username, password)
        if user:
            st.session_state['logged_in'] = True
            st.session_state['user'] = user
            st.success("Logged in successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid username or password")

# Main application
def main_app():
    user = st.session_state['user']
    st.sidebar.title(f"Welcome, {user['full_name']}")
    st.sidebar.subheader(f"Role: {user['role'].capitalize()}")
    
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    
    # Navigation based on user role
    menu_options = ["Dashboard", "Equipment Management", "Maintenance Tracking"]
    
    if user['role'] in ['admin', 'manager']:
        menu_options.extend(["Failure Prediction", "Reporting & Analytics"])
    
    if user['role'] == 'admin':
        menu_options.append("User Management")
    
    selected_page = st.sidebar.selectbox("Navigation", menu_options)
    
    # Page routing
    if selected_page == "Dashboard":
        show_dashboard(user)
    elif selected_page == "Equipment Management":
        equipment_management(user)
    elif selected_page == "Maintenance Tracking":
        maintenance_tracking(user)
    elif selected_page == "Failure Prediction":
        failure_prediction(user)
    elif selected_page == "Reporting & Analytics":
        reporting_analytics(user)
    elif selected_page == "User Management":
        user_management(user)

# Dashboard
def show_dashboard(user):
    st.title("Dashboard")
    
    conn = create_db_connection()
    if not conn:
        return
    
    # Equipment status summary
    st.subheader("Equipment Status")
    col1, col2, col3 = st.columns(3)
    
    equipment_stats = get_equipment_status(conn)
    
    with col1:
        st.metric("Operational", equipment_stats['operational'])
    with col2:
        st.metric("Under Maintenance", equipment_stats['maintenance'])
    with col3:
        st.metric("Failed", equipment_stats['failed'])
    
    # Maintenance alerts
    st.subheader("Upcoming Maintenance")
    upcoming_maintenance = get_upcoming_maintenance(conn, days=7)
    st.dataframe(upcoming_maintenance)
    
    # Failure predictions
    if user['role'] in ['admin', 'manager']:
        st.subheader("High Risk Equipment")
        high_risk = get_high_risk_predictions(conn, threshold=0.7)
        st.dataframe(high_risk)
    
    # Recent sensor data
    st.subheader("Recent Sensor Readings")
    recent_sensor_data = get_recent_sensor_data(conn, limit=10)
    st.dataframe(recent_sensor_data)
    
    conn.close()

def equipment_management(user):
    st.title("Equipment Management")
    
    conn = create_db_connection()
    if not conn:
        st.error("Failed to connect to database")
        return
    
    tab1, tab2, tab3 = st.tabs(["View Equipment", "Add Equipment", "Update Equipment"])
    
    with tab1:
        st.subheader("All Equipment")
        equipment = get_all_equipment(conn)
        
        # Check if equipment dataframe is empty
        if equipment.empty:
            st.warning("No equipment found in database. Please add equipment first.")
        else:
            # Display equipment list
            st.dataframe(equipment)
            
            # Equipment details section
            st.subheader("Equipment Details")
            selected_equipment = st.selectbox(
                "Select equipment to view details", 
                equipment['name'].tolist(),
                index=0,
                key="equipment_details_select"  # Add unique key here
            )
            
            # Safely get equipment ID
            eq_row = equipment[equipment['name'] == selected_equipment]
            if not eq_row.empty:
                try:
                    # Convert to native Python int
                    eq_id = int(eq_row['equipment_id'].iloc[0])
                    eq_details = get_equipment_details(conn, eq_id)
                    
                    if eq_details:
                        # Display equipment details
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Model:** {eq_details.get('model', 'N/A')}")
                            st.write(f"**Serial Number:** {eq_details.get('serial_number', 'N/A')}")
                            st.write(f"**Manufacturer:** {eq_details.get('manufacturer', 'N/A')}")
                        
                        with col2:
                            st.write(f"**Installation Date:** {eq_details.get('installation_date', 'N/A')}")
                            st.write(f"**Last Maintenance:** {eq_details.get('last_maintenance_date', 'N/A')}")
                            st.write(f"**Status:** {eq_details.get('status', 'N/A')}")
                        
                        # Equipment sensor data chart
                        st.subheader("Sensor Data Trends")
                        sensor_data = get_equipment_sensor_data(conn, eq_id, days=30)
                        
                        if not sensor_data.empty:
                            metric = st.selectbox(
                                "Select metric to visualize", 
                                ['temperature', 'vibration', 'pressure', 'current_flow', 'oil_level'],
                                key="metric_select"  # Add unique key here
                            )
                            fig = px.line(
                                sensor_data, 
                                x='timestamp', 
                                y=metric, 
                                title=f"{metric.capitalize()} Trend"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No sensor data available for this equipment")
                    else:
                        st.error("Failed to load equipment details")
                except Exception as e:
                    st.error(f"Error displaying equipment details: {str(e)}")
            else:
                st.warning("Selected equipment not found in database")
    
    # Remove the duplicate selectbox that appears after conn.close()
    # This is the problematic duplicate that needs to be removed:
    # selected_equipment = st.selectbox("Select equipment to view details", 
    #                                  equipment['name'].tolist())
    
    # Rest of your tabs (Add Equipment, Update Equipment)
    with tab2:
        if user['role'] in ['admin', 'manager']:
            st.subheader("Add New Equipment")
            
            with st.form("add_equipment"):
                name = st.text_input("Equipment Name")
                model = st.text_input("Model")
                serial_number = st.text_input("Serial Number")
                manufacturer = st.text_input("Manufacturer")
                installation_date = st.date_input("Installation Date")
                location = st.text_input("Location")
                notes = st.text_area("Notes")
                
                submitted = st.form_submit_button("Add Equipment")
                if submitted:
                    result = add_equipment(conn, name, model, serial_number, manufacturer, 
                                         installation_date, location, notes)
                    if result:
                        st.success("Equipment added successfully!")
                        time.sleep(1)
                        st.rerun()
        else:
            st.warning("You don't have permission to add equipment")
    
    with tab3:
        st.subheader("Update Equipment")
        equipment_list = get_all_equipment(conn)
        selected_equipment = st.selectbox(
            "Select equipment to update", 
            equipment_list['name'].tolist(),
            key="update_equipment_select"  # Add unique key here
        )
        
        if selected_equipment:
            eq_id = equipment_list[equipment_list['name'] == selected_equipment]['equipment_id'].values[0]
            eq_details = get_equipment_details(conn, eq_id)
            
            with st.form("update_equipment"):
                name = st.text_input("Equipment Name", value=eq_details['name'])
                model = st.text_input("Model", value=eq_details['model'])
                serial_number = st.text_input("Serial Number", value=eq_details['serial_number'])
                manufacturer = st.text_input("Manufacturer", value=eq_details['manufacturer'])
                status = st.selectbox("Status", 
                                     ['operational', 'maintenance', 'failed'],
                                     index=['operational', 'maintenance', 'failed'].index(eq_details['status']))
                location = st.text_input("Location", value=eq_details['location'])
                notes = st.text_area("Notes", value=eq_details['notes'])
                
                submitted = st.form_submit_button("Update Equipment")
                if submitted:
                    result = update_equipment(conn, eq_id, name, model, serial_number, 
                                             manufacturer, status, location, notes)
                    if result:
                        st.success("Equipment updated successfully!")
                        time.sleep(1)
                        st.rerun()
    
    conn.close()

# Maintenance Tracking
def maintenance_tracking(user):
    st.title("Maintenance Tracking")
    
    conn = create_db_connection()
    if not conn:
        return
    
    tab1, tab2, tab3 = st.tabs(["View Maintenance", "Schedule Maintenance", "Update Maintenance"])
    
    with tab1:
        st.subheader("Maintenance Activities")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Filter by status", 
                                       ['All', 'scheduled', 'in_progress', 'completed', 'cancelled'])
        with col2:
            type_filter = st.selectbox("Filter by type", 
                                     ['All', 'preventive', 'corrective', 'predictive'])
        with col3:
            time_filter = st.selectbox("Filter by time", 
                                     ['All', 'Past', 'Upcoming'], index=2)
        
        # Get filtered maintenance
        maintenance = get_filtered_maintenance(conn, status_filter, type_filter, time_filter)
        st.dataframe(maintenance)
    
    with tab2:
        st.subheader("Schedule New Maintenance")
        
        equipment = get_all_equipment(conn)
        technicians = get_users_by_role(conn, 'technician')
        
        # Check if we have equipment and technicians
        if equipment.empty:
            st.warning("No equipment available. Please add equipment first.")
        elif not technicians:
            st.warning("No technicians available. Please add technicians first.")
        else:
            with st.form("schedule_maintenance"):
                equipment_id = st.selectbox("Equipment", 
                                          equipment['name'].tolist())
                eq_id = equipment[equipment['name'] == equipment_id]['equipment_id'].values[0]
                
                maintenance_type = st.selectbox("Maintenance Type", 
                                              ['preventive', 'corrective', 'predictive'])
                
                description = st.text_area("Description")
                scheduled_date = st.date_input("Scheduled Date")
                
                # Create technician display strings
                technician_options = [f"{t['full_name']} ({t['username']})" for t in technicians]
                technician = st.selectbox("Assigned Technician", technician_options)
                
                # Safely get technician ID
                try:
                    selected_index = technician_options.index(technician)
                    tech_id = technicians[selected_index]['user_id']
                except ValueError:
                    st.error("Invalid technician selection")
                    tech_id = None
                
                submitted = st.form_submit_button("Schedule Maintenance")
                if submitted and tech_id is not None:
                    result = schedule_maintenance(conn, eq_id, maintenance_type, description, 
                                                 scheduled_date, tech_id)
                    if result:
                        st.success("Maintenance scheduled successfully!")
                        time.sleep(1)
                        st.rerun()
    
    with tab3:
        st.subheader("Update Maintenance Status")
        
        active_maintenance = get_filtered_maintenance(conn, 'All', 'All', 'Upcoming')
        if active_maintenance.empty:
            st.warning("No maintenance activities to update")
        else:
            selected_maintenance = st.selectbox("Select maintenance activity", 
                                              active_maintenance['description'].tolist())
            maint_id = active_maintenance[active_maintenance['description'] == selected_maintenance]['maintenance_id'].values[0]
            
            current_status = active_maintenance[active_maintenance['description'] == selected_maintenance]['status'].values[0]
            
            with st.form("update_maintenance"):
                new_status = st.selectbox("Status", 
                                        ['scheduled', 'in_progress', 'completed', 'cancelled'],
                                        index=['scheduled', 'in_progress', 'completed', 'cancelled'].index(current_status))
                
                completed_date = None
                if new_status == 'completed':
                    completed_date = st.date_input("Completion Date")
                
                duration = st.number_input("Duration (minutes)", min_value=0, value=0)
                cost = st.number_input("Cost", min_value=0.0, value=0.0)
                notes = st.text_area("Notes")
                
                submitted = st.form_submit_button("Update Maintenance")
                if submitted:
                    result = update_maintenance_status(conn, maint_id, new_status, completed_date, duration, cost, notes)
                    if result:
                        st.success("Maintenance updated successfully!")
                        time.sleep(1)
                        st.rerun()
    
    conn.close()
# Failure Prediction
def failure_prediction(user):
    if user['role'] not in ['admin', 'manager']:
        st.warning("You don't have permission to access this page")
        return
    
    st.title("Failure Prediction")
    
    conn = create_db_connection()
    if not conn:
        return
    
    tab1, tab2, tab3 = st.tabs(["Predictions", "Train Model", "Model Performance"])
    
    with tab1:
        st.subheader("Equipment Failure Predictions")
        
        # Get high-risk predictions
        predictions = get_recent_predictions(conn, threshold=0.5)
        st.dataframe(predictions)
        
        # Manual prediction
        st.subheader("Run Prediction for Equipment")
        equipment = get_all_equipment(conn)
        selected_equipment = st.selectbox("Select equipment", 
                                        equipment['name'].tolist())
        eq_id = equipment[equipment['name'] == selected_equipment]['equipment_id'].values[0]
        
        if st.button("Predict Failure Probability"):
            with st.spinner("Running prediction..."):
                # Get recent sensor data
                sensor_data = get_equipment_sensor_data(conn, eq_id, days=7)
                
                if not sensor_data.empty:
                    # Get the latest model for this equipment type
                    model = get_latest_model(conn)
                    
                    if model:
                        # Prepare data for prediction
                        features = sensor_data[['temperature', 'vibration', 'pressure', 'current_flow', 'oil_level']].mean().to_dict()
                        
                        # Predict (in a real app, this would use the actual model)
                        prediction = predict_equipment_failure(features)
                        
                        # Save prediction to database
                        save_prediction(conn, eq_id, model['model_id'], prediction['probability'], 
                                      prediction['failure_type'], prediction['predicted_date'])
                        
                        # Display results
                        st.subheader("Prediction Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Failure Probability", f"{prediction['probability']*100:.2f}%")
                        with col2:
                            st.metric("Likely Failure Type", prediction['failure_type'])
                        with col3:
                            st.metric("Predicted Date", prediction['predicted_date'])
                        
                        # Show feature importance
                        st.subheader("Key Contributing Factors")
                        fig = px.bar(x=list(prediction['feature_importance'].keys()),
                                    y=list(prediction['feature_importance'].values()),
                                    labels={'x': 'Feature', 'y': 'Importance'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No prediction model available for this equipment type")
                else:
                    st.error("No sensor data available for prediction")
    
    with tab2:
        st.subheader("Train New Prediction Model")
        
        equipment_types = get_equipment_types(conn)
        selected_type = st.selectbox("Select equipment type", equipment_types)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Get historical data for this equipment type
                historical_data = get_historical_training_data(conn, selected_type)
                
                if not historical_data.empty:
                    # Train model (in a real app, this would do actual training)
                    model_info = train_failure_prediction_model(historical_data, selected_type)
                    
                    # Save model to database
                    save_model(conn, model_info['name'], selected_type, model_info['path'], 
                              model_info['accuracy'])
                    
                    st.success(f"Model trained successfully! Accuracy: {model_info['accuracy']*100:.2f}%")
                else:
                    st.error("Not enough historical data available for training")
    
    with tab3:
        st.subheader("Model Performance")
        
        models = get_all_models(conn)
        st.dataframe(models)
        
        if not models.empty:
            selected_model = st.selectbox("Select model to view details", 
                                       models['model_name'].tolist())
            model_id = models[models['model_name'] == selected_model]['model_id'].values[0]
            
            # Get model predictions
            predictions = get_model_predictions(conn, model_id)
            
            if not predictions.empty:
                # Calculate accuracy metrics
                verified = predictions[predictions['is_verified'] == True]
                if not verified.empty:
                    accuracy = verified['actual_outcome'].mean()
                    precision = verified[verified['failure_probability'] > 0.5]['actual_outcome'].mean()
                    recall = verified[verified['actual_outcome'] == True]['failure_probability'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{accuracy*100:.2f}%")
                    with col2:
                        st.metric("Precision", f"{precision*100:.2f}%" if not pd.isna(precision) else "N/A")
                    with col3:
                        st.metric("Recall", f"{recall*100:.2f}%" if not pd.isna(recall) else "N/A")
                
                # ROC curve (simulated)
                st.subheader("Model Performance Metrics")
                st.line_chart(pd.DataFrame({
                    'False Positive Rate': np.linspace(0, 1, 100),
                    'True Positive Rate': np.linspace(0, 1, 100)**0.5
                }))
    
    conn.close()

# Reporting & Analytics
def reporting_analytics(user):
    if user['role'] not in ['admin', 'manager']:
        st.warning("You don't have permission to access this page")
        return
    
    st.title("Reporting & Analytics")
    
    conn = create_db_connection()
    if not conn:
        return
    
    tab1, tab2, tab3 = st.tabs(["Maintenance Reports", "Failure Analysis", "Equipment Performance"])
    
    with tab1:
        st.subheader("Maintenance Reports")
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                # Get maintenance data
                maintenance = get_maintenance_report(conn, start_date, end_date)
                
                if not maintenance.empty:
                    # Summary metrics
                    st.subheader("Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    total_maintenance = len(maintenance)
                    completed = maintenance[maintenance['status'] == 'completed']
                    total_cost = completed['cost'].sum()
                    avg_duration = completed['duration_minutes'].mean()
                    
                    with col1:
                        st.metric("Total Maintenance Activities", total_maintenance)
                    with col2:
                        st.metric("Total Cost", f"${total_cost:,.2f}")
                    with col3:
                        st.metric("Average Duration", f"{avg_duration:.1f} minutes" if not pd.isna(avg_duration) else "N/A")
                    
                    # Maintenance by type
                    st.subheader("Maintenance by Type")
                    type_counts = maintenance['maintenance_type'].value_counts()
                    fig1 = px.pie(type_counts, values=type_counts.values, names=type_counts.index)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Cost over time
                    st.subheader("Maintenance Cost Over Time")
                    completed['completed_date'] = pd.to_datetime(completed['completed_date'])
                    cost_over_time = completed.groupby(completed['completed_date'].dt.date)['cost'].sum().reset_index()
                    fig2 = px.line(cost_over_time, x='completed_date', y='cost')
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Detailed data
                    st.subheader("Detailed Data")
                    st.dataframe(maintenance)
                else:
                    st.warning("No maintenance data found for the selected period")
    
    with tab2:
        st.subheader("Failure Analysis")
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Failure Start Date", datetime.now() - timedelta(days=90))
        with col2:
            end_date = st.date_input("Failure End Date", datetime.now())
        
        if st.button("Analyze Failures"):
            with st.spinner("Analyzing failure data..."):
                # Get failure data
                failures = get_failure_report(conn, start_date, end_date)
                
                if not failures.empty:
                    # Summary metrics
                    st.subheader("Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    total_failures = len(failures)
                    total_downtime = failures['downtime_minutes'].sum()
                    avg_repair_cost = failures['repair_cost'].mean()
                    
                    with col1:
                        st.metric("Total Failures", total_failures)
                    with col2:
                        st.metric("Total Downtime", f"{total_downtime/60:.1f} hours")
                    with col3:
                        st.metric("Average Repair Cost", f"${avg_repair_cost:,.2f}")
                    
                    # Failure types
                    st.subheader("Failure Types")
                    type_counts = failures['failure_type'].value_counts()
                    fig1 = px.bar(type_counts, x=type_counts.index, y=type_counts.values)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Downtime by equipment
                    st.subheader("Downtime by Equipment")
                    downtime_by_eq = failures.groupby('equipment_id')['downtime_minutes'].sum().reset_index()
                    fig2 = px.bar(downtime_by_eq, x='equipment_id', y='downtime_minutes')
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Detailed data
                    st.subheader("Detailed Data")
                    st.dataframe(failures)
                else:
                    st.warning("No failure data found for the selected period")
    
    with tab3:
        st.subheader("Equipment Performance")
        
        equipment = get_all_equipment(conn)
        selected_equipment = st.selectbox("Select equipment", 
                                        equipment['name'].tolist())
        eq_id = equipment[equipment['name'] == selected_equipment]['equipment_id'].values[0]
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Performance Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("Performance End Date", datetime.now())
        
        if st.button("Generate Performance Report"):
            with st.spinner("Generating performance report..."):
                # Get equipment performance data
                performance = get_equipment_performance(conn, eq_id, start_date, end_date)
                
                if not performance.empty:
                    # Uptime/downtime
                    st.subheader("Uptime/Downtime")
                    total_hours = (end_date - start_date).days * 24
                    uptime_hours = total_hours - (performance['downtime_minutes'].sum() / 60)
                    uptime_percent = (uptime_hours / total_hours) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Uptime", f"{uptime_hours:.1f} hours")
                    with col2:
                        st.metric("Uptime Percentage", f"{uptime_percent:.1f}%")
                    
                    # Maintenance vs failure
                    st.subheader("Maintenance vs Failure")
                    maint_count = performance['maintenance_count'].values[0]
                    failure_count = performance['failure_count'].values[0]
                    
                    fig1 = px.pie(values=[maint_count, failure_count], 
                                names=['Maintenance', 'Failures'])
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Cost breakdown
                    st.subheader("Cost Breakdown")
                    maint_cost = performance['maintenance_cost'].values[0]
                    repair_cost = performance['repair_cost'].values[0]
                    
                    fig2 = px.pie(values=[maint_cost, repair_cost], 
                                names=['Maintenance', 'Repairs'])
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No performance data found for the selected equipment and period")
    
    conn.close()

# User Management
def user_management(user):
    if user['role'] != 'admin':
        st.warning("You don't have permission to access this page")
        return
    
    st.title("User Management")
    
    conn = create_db_connection()
    if not conn:
        return
    
    tab1, tab2, tab3 = st.tabs(["View Users", "Add User", "Update User"])
    
    with tab1:
        st.subheader("All Users")
        users = get_all_users(conn)
        st.dataframe(users)
    
    with tab2:
        st.subheader("Add New User")
        
        with st.form("add_user"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            role = st.selectbox("Role", ['admin', 'manager', 'technician'])
            
            submitted = st.form_submit_button("Add User")
            if submitted:
                # Hash password (in a real app, use proper password hashing)
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                
                result = add_user(conn, username, password_hash, full_name, email, role)
                if result:
                    st.success("User added successfully!")
                    time.sleep(1)
                    st.rerun()
    
    with tab3:
        st.subheader("Update User")
        
        users = get_all_users(conn)
        selected_user = st.selectbox("Select user to update", 
                                   users['username'].tolist())
        
        if selected_user:
            # Convert numpy.int64 to native Python int
            user_id = int(users[users['username'] == selected_user]['user_id'].values[0])
            user_details = get_user_details(conn, user_id)
            
            with st.form("update_user"):
                username = st.text_input("Username", value=user_details['username'])
                full_name = st.text_input("Full Name", value=user_details['full_name'])
                email = st.text_input("Email", value=user_details['email'])
                role = st.selectbox("Role", 
                                   ['admin', 'manager', 'technician'],
                                   index=['admin', 'manager', 'technician'].index(user_details['role']))
                
                submitted = st.form_submit_button("Update User")
                if submitted:
                    result = update_user(conn, user_id, username, full_name, email, role)
                    if result:
                        st.success("User updated successfully!")
                        time.sleep(1)
                        st.rerun()
    
    conn.close()

# Main application flow
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        login_page()
    else:
        main_app()