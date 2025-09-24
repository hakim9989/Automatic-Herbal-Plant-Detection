<?php
// get_history.php
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

// ✅ Validate the token and extend expiry if valid
$user_id = null;
$validateTokenSql = "SELECT user_id, expires_at FROM auth_tokens WHERE auth_token = ?";
if ($stmt = $conn->prepare($validateTokenSql)) {
    $stmt->bind_param("s", $authToken);
    $stmt->execute();
    $stmt->store_result();

    if ($stmt->num_rows == 1) {
        $stmt->bind_result($db_user_id, $expires_at);
        $stmt->fetch();

        $now = new DateTime();
        $expiry = $expires_at ? new DateTime($expires_at) : null;

        if ($expiry === null || $now < $expiry) {
            $user_id = $db_user_id;

            // ✅ Extend expiry (sliding session)
            $newExpiry = date('Y-m-d H:i:s', strtotime('+24 hours'));
            $updateSql = "UPDATE auth_tokens SET expires_at = ? WHERE auth_token = ?";
            if ($updateStmt = $conn->prepare($updateSql)) {
                $updateStmt->bind_param("ss", $newExpiry, $authToken);
                $updateStmt->execute();
                $updateStmt->close();
            }
        } else {
            // ❌ Token expired → delete it
            $deleteSql = "DELETE FROM auth_tokens WHERE auth_token = ?";
            if ($delStmt = $conn->prepare($deleteSql)) {
                $delStmt->bind_param("s", $authToken);
                $delStmt->execute();
                $delStmt->close();
            }
            http_response_code(401);
            echo json_encode(["success" => false, "message" => "Token expired. Please log in again."]);
            exit();
        }
    } else {
        http_response_code(401);
        echo json_encode(["success" => false, "message" => "Invalid authentication token."]);
        exit();
    }
    $stmt->close();
} else {
    http_response_code(500);
    echo json_encode(["success" => false, "message" => "Database error: " . $conn->error]);
    exit();
}

// ✅ Only allow GET
if ($_SERVER["REQUEST_METHOD"] !== "GET") {
    http_response_code(405);
    echo json_encode(["success" => false, "message" => "Invalid request method."]);
    exit();
}

// ✅ Retrieve history
$sql = "SELECT id, plant_type, confidence, scientific_name, description, medicinal_uses, timestamp 
        FROM plant_detections 
        WHERE user_id = ? 
        ORDER BY timestamp DESC";

if ($stmt = $conn->prepare($sql)) {
    $stmt->bind_param("i", $user_id);
    $stmt->execute();
    $result = $stmt->get_result();

    $detections = [];
    while ($row = $result->fetch_assoc()) {
        $row['confidence'] = floatval($row['confidence']);
        $detections[] = $row;
    }

    echo json_encode(["success" => true, "detections" => $detections]);

    $stmt->close();
} else {
    http_response_code(500);
    echo json_encode(["success" => false, "message" => "Database error: " . $conn->error]);
}

$conn->close();
