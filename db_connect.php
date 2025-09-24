<?php
// db_connect.php

// --- CORS (Cross-Origin Resource Sharing) Headers ---
// Allow cross-origin requests from any origin (ideal for development, but consider restricting in production)
header("Access-Control-Allow-Origin: *");
// Allow specific HTTP methods, including POST, GET, and critically, OPTIONS for preflight requests
header("Access-Control-Allow-Methods: GET, POST, OPTIONS, PUT, DELETE");
// Allow specific headers that the client might send, like Content-Type for JSON and Authorization for tokens
header("Access-Control-Allow-Headers: Content-Type, Authorization, X-Auth-Token");
// Optional: Cache the preflight response for 24 hours (86400 seconds) to reduce the number of OPTIONS requests
header("Access-Control-Max-Age: 86400");

// --- Handle Preflight OPTIONS Request ---
// A browser sends an OPTIONS request before the actual request (e.g., POST, PUT) to check
// if the server understands CORS and if the actual request is safe to send.
// We must handle this by sending back the CORS headers and a 200 OK status, then exiting.
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200); // Respond with 200 OK
    exit(); // Terminate script execution; no further processing is needed for an OPTIONS request
}

// --- Database Configuration ---
// Replace with your actual database credentials
define('DB_SERVER', 'localhost');
define('DB_USERNAME', 'root'); // Default XAMPP username
define('DB_PASSWORD', '');     // Default XAMPP password ()
define('DB_NAME', 'plant_detection_db'); // Your database name

// --- Establish Connection ---
// Create a new mysqli object to connect to the database
$conn = new mysqli(DB_SERVER, DB_USERNAME, DB_PASSWORD, DB_NAME);

// --- Check Connection ---
// If the connection fails, send a server error response and stop the script.
if ($conn->connect_error) {
    http_response_code(500); // Internal Server Error
    header('Content-Type: application/json'); // Ensure client knows to expect JSON
    // Provide a clear error message in a JSON format
    echo json_encode(["success" => false, "message" => "Database connection failed: " . $conn->connect_error]);
    exit();
}

// Set the character set to utf8mb4 for full Unicode support
$conn->set_charset("utf8mb4");

?>
