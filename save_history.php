<?php
// save_history.php
session_start();
header('Content-Type: application/json');
require_once 'db_connect.php';

$headers = getallheaders();
$authToken = $headers['X-Auth-Token'] ?? null;

if (!$authToken) {
    http_response_code(401);
    echo json_encode(["success" => false, "message" => "Authentication required."]);
    exit();
}

// Validate the token
$user_id = null;
$validateTokenSql = "SELECT user_id, expires_at FROM auth_tokens WHERE auth_token = ?";
if ($stmt = $conn->prepare($validateTokenSql)) {
    $stmt->bind_param("s", $authToken);
    $stmt->execute();
    $stmt->store_result();

    if ($stmt->num_rows === 1) {
        $stmt->bind_result($db_user_id, $expires_at);
        $stmt->fetch();

        if ($expires_at === null || new DateTime() < new DateTime($expires_at)) {
            $user_id = $db_user_id;
        } else {
            http_response_code(401);
            echo json_encode(["success" => false, "message" => "Token expired. Please log in again."]);
            exit();
        }
    } else {
        http_response_code(401);
        echo json_encode(["success" => false, "message" => "Invalid token."]);
        exit();
    }
    $stmt->close();
} else {
    http_response_code(500);
    echo json_encode(["success" => false, "message" => "Database error (auth): " . $conn->error]);
    exit();
}

// Read JSON input
$input = json_decode(file_get_contents("php://input"), true);
$plant_type = $input['plant_type'] ?? '';
$confidence = $input['confidence'] ?? '';

$details = $input['details'] ?? [];
$scientific_name = $details['scientific_name'] ?? '';
$description = $details['description'] ?? '';
$medicinal_uses = $details['medicinal_uses'] ?? '';

if (empty($plant_type) || empty($confidence)) {
    http_response_code(400);
    echo json_encode(["success" => false, "message" => "Required fields are missing."]);
    exit();
}

// Save detection result
$insertSql = "INSERT INTO plant_detections (
    user_id, plant_type, confidence, scientific_name, description, medicinal_uses, timestamp
) VALUES (?, ?, ?, ?, ?, ?, NOW())";

if ($stmt = $conn->prepare($insertSql)) {
    $stmt->bind_param(
        "isdsss",
        $user_id,
        $plant_type,
        $confidence,
        $scientific_name,
        $description,
        $medicinal_uses
    );

    if ($stmt->execute()) {
        echo json_encode(["success" => true, "message" => "Detection saved successfully."]);
    } else {
        http_response_code(500);
        echo json_encode(["success" => false, "message" => "Failed to save detection: " . $stmt->error]);
    }

    $stmt->close();
} else {
    http_response_code(500);
    echo json_encode(["success" => false, "message" => "Database error (insert): " . $conn->error]);
}

$conn->close();
