import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime

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
        print(f"Error connecting to MySQL database: {e}")
        return None

# Equipment operations
def get_all_equipment(conn):
    query = "SELECT * FROM equipment"
    df = pd.read_sql(query, conn)
    # Convert numpy types to native Python types
    if not df.empty and 'equipment_id' in df.columns:
        df['equipment_id'] = df['equipment_id'].apply(lambda x: int(x) if pd.notnull(x) else x)
    return df

def get_equipment_details(conn, equipment_id):
    query = "SELECT * FROM equipment WHERE equipment_id = %s"
    cursor = conn.cursor(dictionary=True)
    try:
        # Convert to native Python int if it's numpy type
        equipment_id = int(equipment_id) if hasattr(equipment_id, 'item') else equipment_id
        cursor.execute(query, (equipment_id,))
        return cursor.fetchone()
    except Error as e:
        print(f"Error fetching equipment details: {e}")
        return None

def get_equipment_status(conn):
    query = """
    SELECT status, COUNT(*) as count 
    FROM equipment 
    GROUP BY status
    """
    result = pd.read_sql(query, conn)
    status_counts = {'operational': 0, 'maintenance': 0, 'failed': 0}
    for _, row in result.iterrows():
        status_counts[row['status']] = row['count']
    return status_counts
def get_equipment_sensor_data(conn, equipment_id, days=7):
    query = """
    SELECT * FROM sensor_data 
    WHERE equipment_id = %s 
    AND timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
    ORDER BY timestamp DESC
    """
    return pd.read_sql(query, conn, params=(equipment_id, days))

def add_equipment(conn, name, model, serial_number, manufacturer, installation_date, location, notes):
    query = """
    INSERT INTO equipment 
    (name, model, serial_number, manufacturer, installation_date, location, notes)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (name, model, serial_number, manufacturer, installation_date, location, notes))
        conn.commit()
        return True
    except Error as e:
        print(f"Error adding equipment: {e}")
        conn.rollback()
        return False

def update_equipment(conn, equipment_id, name, model, serial_number, manufacturer, status, location, notes):
    query = """
    UPDATE equipment 
    SET name = %s, model = %s, serial_number = %s, manufacturer = %s, 
        status = %s, location = %s, notes = %s
    WHERE equipment_id = %s
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (name, model, serial_number, manufacturer, status, location, notes, equipment_id))
        conn.commit()
        return True
    except Error as e:
        print(f"Error updating equipment: {e}")
        conn.rollback()
        return False

# Maintenance operations
def get_upcoming_maintenance(conn, days=7):
    query = """
    SELECT m.maintenance_id, e.name as equipment, m.maintenance_type, 
           m.scheduled_date, m.status, u.full_name as assigned_to
    FROM maintenance m
    JOIN equipment e ON m.equipment_id = e.equipment_id
    LEFT JOIN users u ON m.performed_by = u.user_id
    WHERE m.scheduled_date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL %s DAY)
    ORDER BY m.scheduled_date
    """
    return pd.read_sql(query, conn, params=(days,))

def get_filtered_maintenance(conn, status_filter, type_filter, time_filter):
    query = """
    SELECT m.maintenance_id, e.name as equipment, m.maintenance_type, 
           m.description, m.scheduled_date, m.completed_date, 
           m.status, m.duration_minutes, m.cost, u.full_name as performed_by
    FROM maintenance m
    JOIN equipment e ON m.equipment_id = e.equipment_id
    LEFT JOIN users u ON m.performed_by = u.user_id
    WHERE 1=1
    """
    
    params = []
    
    if status_filter != 'All':
        query += " AND m.status = %s"
        params.append(status_filter)
    
    if type_filter != 'All':
        query += " AND m.maintenance_type = %s"
        params.append(type_filter)
    
    if time_filter == 'Past':
        query += " AND m.scheduled_date < CURDATE()"
    elif time_filter == 'Upcoming':
        query += " AND m.scheduled_date >= CURDATE()"
    
    query += " ORDER BY m.scheduled_date"
    
    return pd.read_sql(query, conn, params=params)

def schedule_maintenance(conn, equipment_id, maintenance_type, description, scheduled_date, performed_by):
    query = """
    INSERT INTO maintenance 
    (equipment_id, maintenance_type, description, scheduled_date, performed_by)
    VALUES (%s, %s, %s, %s, %s)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (equipment_id, maintenance_type, description, scheduled_date, performed_by))
        conn.commit()
        return True
    except Error as e:
        print(f"Error scheduling maintenance: {e}")
        conn.rollback()
        return False

def update_maintenance_status(conn, maintenance_id, status, completed_date=None, duration=0, cost=0, notes=None):
    query = """
    UPDATE maintenance 
    SET status = %s, completed_date = %s, duration_minutes = %s, cost = %s
    WHERE maintenance_id = %s
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (status, completed_date, duration, cost, maintenance_id))
        conn.commit()
        return True
    except Error as e:
        print(f"Error updating maintenance: {e}")
        conn.rollback()
        return False

def get_maintenance_report(conn, start_date, end_date):
    query = """
    SELECT m.maintenance_id, e.name as equipment, m.maintenance_type, 
           m.description, m.scheduled_date, m.completed_date, 
           m.status, m.duration_minutes, m.cost, u.full_name as performed_by
    FROM maintenance m
    JOIN equipment e ON m.equipment_id = e.equipment_id
    LEFT JOIN users u ON m.performed_by = u.user_id
    WHERE (m.scheduled_date BETWEEN %s AND %s)
       OR (m.completed_date BETWEEN %s AND %s)
    ORDER BY m.scheduled_date
    """
    return pd.read_sql(query, conn, params=(start_date, end_date, start_date, end_date))

# Failure operations
def get_high_risk_predictions(conn, threshold=0.7):
    query = """
    SELECT p.prediction_id, e.name as equipment, p.failure_probability, 
           p.predicted_failure_type, p.predicted_failure_date, 
           pm.model_name, p.prediction_date
    FROM predictions p
    JOIN equipment e ON p.equipment_id = e.equipment_id
    JOIN prediction_models pm ON p.model_id = pm.model_id
    WHERE p.failure_probability >= %s
    AND (p.is_verified IS NULL OR p.is_verified = FALSE)
    ORDER BY p.failure_probability DESC
    """
    return pd.read_sql(query, conn, params=(threshold,))

def get_failure_report(conn, start_date, end_date):
    query = """
    SELECT f.failure_id, e.name as equipment, f.failure_type, 
           f.failure_date, f.downtime_minutes, f.repair_cost,
           f.root_cause, u.full_name as resolved_by
    FROM failure_records f
    JOIN equipment e ON f.equipment_id = e.equipment_id
    LEFT JOIN users u ON f.resolved_by = u.user_id
    WHERE f.failure_date BETWEEN %s AND %s
    ORDER BY f.failure_date DESC
    """
    return pd.read_sql(query, conn, params=(start_date, end_date))

# Prediction operations
def get_recent_predictions(conn, threshold=0.5):
    query = """
    SELECT p.prediction_id, e.name as equipment, p.failure_probability, 
           p.predicted_failure_type, p.predicted_failure_date, 
           p.prediction_date, p.is_verified, p.actual_outcome
    FROM predictions p
    JOIN equipment e ON p.equipment_id = e.equipment_id
    WHERE p.failure_probability >= %s
    ORDER BY p.prediction_date DESC
    LIMIT 50
    """
    return pd.read_sql(query, conn, params=(threshold,))

def get_latest_model(conn):
    query = """
    SELECT * FROM prediction_models
    ORDER BY last_updated DESC
    LIMIT 1
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    return cursor.fetchone()

def save_prediction(conn, equipment_id, model_id, probability, failure_type, predicted_date):
    query = """
    INSERT INTO predictions 
    (equipment_id, model_id, prediction_date, failure_probability, 
     predicted_failure_type, predicted_failure_date)
    VALUES (%s, %s, NOW(), %s, %s, %s)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (equipment_id, model_id, probability, failure_type, predicted_date))
        conn.commit()
        return True
    except Error as e:
        print(f"Error saving prediction: {e}")
        conn.rollback()
        return False

def get_historical_training_data(conn, equipment_type):
    query = """
    SELECT sd.*, 
           CASE WHEN fr.failure_id IS NOT NULL THEN 1 ELSE 0 END as failure
    FROM sensor_data sd
    JOIN equipment e ON sd.equipment_id = e.equipment_id
    LEFT JOIN failure_records fr ON sd.equipment_id = fr.equipment_id 
        AND ABS(TIMESTAMPDIFF(MINUTE, sd.timestamp, fr.failure_date)) <= 60
    WHERE e.model LIKE %s
    """
    return pd.read_sql(query, conn, params=(f"%{equipment_type}%",))

def save_model(conn, name, equipment_type, path, accuracy):
    query = """
    INSERT INTO prediction_models 
    (model_name, equipment_type, model_path, accuracy_score)
    VALUES (%s, %s, %s, %s)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (name, equipment_type, path, accuracy))
        conn.commit()
        return True
    except Error as e:
        print(f"Error saving model: {e}")
        conn.rollback()
        return False

def get_all_models(conn):
    query = "SELECT * FROM prediction_models ORDER BY last_updated DESC"
    return pd.read_sql(query, conn)

def get_model_predictions(conn, model_id):
    query = """
    SELECT p.*, e.name as equipment
    FROM predictions p
    JOIN equipment e ON p.equipment_id = e.equipment_id
    WHERE p.model_id = %s
    ORDER BY p.prediction_date DESC
    """
    return pd.read_sql(query, conn, params=(model_id,))

# User operations
def get_all_users(conn):
    query = "SELECT user_id, username, full_name, email, role FROM users"
    return pd.read_sql(query, conn)

def get_user_details(conn, user_id):
    query = "SELECT * FROM users WHERE user_id = %s"
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, (user_id,))
    return cursor.fetchone()

def get_users_by_role(conn, role):
    query = "SELECT * FROM users WHERE role = %s"
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, (role,))
    return cursor.fetchall()

def add_user(conn, username, password_hash, full_name, email, role):
    query = """
    INSERT INTO users 
    (username, password_hash, full_name, email, role)
    VALUES (%s, %s, %s, %s, %s)
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (username, password_hash, full_name, email, role))
        conn.commit()
        return True
    except Error as e:
        print(f"Error adding user: {e}")
        conn.rollback()
        return False

def update_user(conn, user_id, username, full_name, email, role):
    query = """
    UPDATE users 
    SET username = %s, full_name = %s, email = %s, role = %s
    WHERE user_id = %s
    """
    cursor = conn.cursor()
    try:
        cursor.execute(query, (username, full_name, email, role, user_id))
        conn.commit()
        return True
    except Error as e:
        print(f"Error updating user: {e}")
        conn.rollback()
        return False

# Equipment performance
def get_equipment_performance(conn, equipment_id, start_date, end_date):
    query = """
    SELECT 
        (SELECT COUNT(*) FROM maintenance 
         WHERE equipment_id = %s 
         AND (scheduled_date BETWEEN %s AND %s)) as maintenance_count,
        (SELECT COUNT(*) FROM failure_records 
         WHERE equipment_id = %s 
         AND (failure_date BETWEEN %s AND %s)) as failure_count,
        (SELECT SUM(downtime_minutes) FROM failure_records 
         WHERE equipment_id = %s 
         AND (failure_date BETWEEN %s AND %s)) as downtime_minutes,
        (SELECT SUM(cost) FROM maintenance 
         WHERE equipment_id = %s 
         AND status = 'completed'
         AND (completed_date BETWEEN %s AND %s)) as maintenance_cost,
        (SELECT SUM(repair_cost) FROM failure_records 
         WHERE equipment_id = %s 
         AND (failure_date BETWEEN %s AND %s)) as repair_cost
    """
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, (equipment_id, start_date, end_date,
                              equipment_id, start_date, end_date,
                              equipment_id, start_date, end_date,
                              equipment_id, start_date, end_date,
                              equipment_id, start_date, end_date))
    return pd.DataFrame([cursor.fetchone()])

def get_equipment_types(conn):
    query = "SELECT DISTINCT model FROM equipment"
    df = pd.read_sql(query, conn)
    return df['model'].dropna().tolist()

def get_recent_sensor_data(conn, limit=10):
    query = """
    SELECT e.name as equipment, sd.timestamp, sd.temperature, 
           sd.vibration, sd.pressure, sd.current_flow, sd.oil_level
    FROM sensor_data sd
    JOIN equipment e ON sd.equipment_id = e.equipment_id
    ORDER BY sd.timestamp DESC
    LIMIT %s
    """
    return pd.read_sql(query, conn, params=(limit,))