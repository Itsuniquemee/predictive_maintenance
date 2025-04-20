-- Create database
CREATE DATABASE IF NOT EXISTS predictive_maintenance;
USE predictive_maintenance;

-- User table for authentication and authorization
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    role ENUM('admin', 'manager', 'technician') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Equipment table
CREATE TABLE equipment (
    equipment_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model VARCHAR(100),
    serial_number VARCHAR(100) UNIQUE,
    manufacturer VARCHAR(100),
    installation_date DATE,
    last_maintenance_date DATE,
    status ENUM('operational', 'maintenance', 'failed') DEFAULT 'operational',
    location VARCHAR(100),
    notes TEXT
);

-- Maintenance activities
CREATE TABLE maintenance (
    maintenance_id INT AUTO_INCREMENT PRIMARY KEY,
    equipment_id INT NOT NULL,
    maintenance_type ENUM('preventive', 'corrective', 'predictive') NOT NULL,
    description TEXT,
    scheduled_date DATE NOT NULL,
    completed_date DATE,
    performed_by INT,
    status ENUM('scheduled', 'in_progress', 'completed', 'cancelled') DEFAULT 'scheduled',
    duration_minutes INT,
    cost DECIMAL(10,2),
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id),
    FOREIGN KEY (performed_by) REFERENCES users(user_id)
);

-- Sensor data
CREATE TABLE sensor_data (
    data_id INT AUTO_INCREMENT PRIMARY KEY,
    equipment_id INT NOT NULL,
    timestamp DATETIME NOT NULL,
    temperature DECIMAL(10,2),
    vibration DECIMAL(10,2),
    pressure DECIMAL(10,2),
    current_flow DECIMAL(10,2),
    oil_level DECIMAL(10,2),
    other_metrics JSON,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
);

-- Failure records
CREATE TABLE failure_records (
    failure_id INT AUTO_INCREMENT PRIMARY KEY,
    equipment_id INT NOT NULL,
    failure_date DATETIME NOT NULL,
    failure_type VARCHAR(100) NOT NULL,
    description TEXT,
    downtime_minutes INT,
    repair_cost DECIMAL(10,2),
    root_cause TEXT,
    resolved_by INT,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id),
    FOREIGN KEY (resolved_by) REFERENCES users(user_id)
);

-- Predictive maintenance models
CREATE TABLE prediction_models (
    model_id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    equipment_type VARCHAR(100) NOT NULL,
    model_path VARCHAR(255) NOT NULL,
    accuracy_score DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Predictions
CREATE TABLE predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    equipment_id INT NOT NULL,
    model_id INT NOT NULL,
    prediction_date DATETIME NOT NULL,
    failure_probability DECIMAL(5,2) NOT NULL,
    predicted_failure_type VARCHAR(100),
    predicted_failure_date DATE,
    confidence_level DECIMAL(5,2),
    is_verified BOOLEAN DEFAULT FALSE,
    actual_outcome BOOLEAN,
    notes TEXT,
    FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id),
    FOREIGN KEY (model_id) REFERENCES prediction_models(model_id)
);

-- Create indexes for better performance
CREATE INDEX idx_sensor_data_equipment ON sensor_data(equipment_id);
CREATE INDEX idx_sensor_data_timestamp ON sensor_data(timestamp);
CREATE INDEX idx_maintenance_equipment ON maintenance(equipment_id);
CREATE INDEX idx_maintenance_status ON maintenance(status);