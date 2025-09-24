<?php
// register.php
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Headers: Content-Type, Authorization");
header("Access-Control-Allow-Methods: GET, POST, OPTIONS");

if ($_SERVER['REQUEST_METHOD'] == 'OPTIONS') {
    http_response_code(200);
    exit();
}

session_start();
header('Content-Type: application/json');
require_once 'db_connect.php';
// ... rest of your code ...

header('Content-Type: application/json');
require_once 'db_connect.php'; // Include database connection

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Get raw POST data
    $input = file_get_contents("php://input");
    $data = json_decode($input, true);

    $userName = $data['userName'] ?? ''; 
    $password = $data['password'] ?? '';

    if (empty($userName) || empty($password)) {
        echo json_encode(["success" => false, "message" => "Username and password are required."]);
        exit();
    }

    // Hash the password securely
    $password_hash = password_hash($password, PASSWORD_DEFAULT);

    // Prepare an insert statement (changed 'email' to 'userName')
    $sql = "INSERT INTO users (userName, password_hash) VALUES (?, ?)";

    if ($stmt = $conn->prepare($sql)) {
        $stmt->bind_param("ss", $userName, $password_hash); // Bind userName

        if ($stmt->execute()) {
            echo json_encode(["success" => true, "message" => "Registration successful!"]);
        } else {
            // Check for duplicate username error (MySQL error code 1062)
            if ($conn->errno == 1062) {
                echo json_encode(["success" => false, "message" => "Username already taken. Please login or use a different username."]);
            } else {
                echo json_encode(["success" => false, "message" => "Error during registration: " . $stmt->error]);
            }
        }
        $stmt->close();
    } else {
        echo json_encode(["success" => false, "message" => "Database query preparation failed: " . $conn->error]);
    }

    $conn->close();
} else {
    echo json_encode(["success" => false, "message" => "Invalid request method."]);
}
if (!preg_match('/^[a-zA-Z0-9]{5,20}$/', $userName)) {
  echo json_encode(["success" => false, "message" => "Username must be 5-20 letters/numbers."]);
  exit();
}

if (strlen($password) < 6) {
  echo json_encode(["success" => false, "message" => "Password must be at least 6 characters."]);
  exit();
}

// NO CLOSING PHP TAG HERE TO PREVENT ACCIDENTAL WHITESPACE
