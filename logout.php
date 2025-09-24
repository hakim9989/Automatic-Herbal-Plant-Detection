<?php
session_start();
header('Content-Type: application/json');
require_once 'db_connect.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(["success" => false, "message" => "Invalid request method. Use POST."]);
    exit();
}

$headers = getallheaders();
$authToken = $headers['X-Auth-Token'] ?? null;
error_log("LOGOUT REQUEST: Received X-Auth-Token: " . ($authToken ?? 'NULL'));

if ($authToken) {
    $deleteTokenSql = "DELETE FROM auth_tokens WHERE auth_token = ?";
    if ($stmt = $conn->prepare($deleteTokenSql)) {
        $stmt->bind_param("s", $authToken);
        if ($stmt->execute()) {
            error_log("LOGOUT SUCCESS: Token deleted from DB: $authToken");
        } else {
            error_log("LOGOUT ERROR: Failed to delete token from DB: " . $stmt->error);
        }
        $stmt->close();
    } else {
        error_log("LOGOUT ERROR: Failed to prepare token delete statement: " . $conn->error);
    }
}

// Clear PHP session
$_SESSION = [];
session_unset();
session_destroy();

$conn->close();
echo json_encode(["success" => true, "message" => "Logged out successfully."]);
// NO CLOSING PHP TAG
