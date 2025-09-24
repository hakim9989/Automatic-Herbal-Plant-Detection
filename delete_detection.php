<?php
require_once 'db_connect.php';
header('Content-Type: application/json');
$headers = getallheaders();
$authToken = $headers['X-Auth-Token'] ?? null;
$data = json_decode(file_get_contents("php://input"), true);
$id = $data['id'] ?? null;

if (!$authToken || !$id) {
    echo json_encode(["success" => false, "message" => "Invalid request."]);
    exit();
}

// Validate token
$stmt = $conn->prepare("SELECT user_id FROM auth_tokens WHERE auth_token = ? AND (expires_at IS NULL OR expires_at > NOW())");
$stmt->bind_param("s", $authToken);
$stmt->execute();
$stmt->store_result();

if ($stmt->num_rows !== 1) {
    echo json_encode(["success" => false, "message" => "Unauthorized."]);
    exit();
}
$stmt->bind_result($user_id);
$stmt->fetch();
$stmt->close();

// Delete detection
$delete = $conn->prepare("DELETE FROM plant_detections WHERE id = ? AND user_id = ?");
$delete->bind_param("ii", $id, $user_id);
if ($delete->execute()) {
    echo json_encode(["success" => true, "message" => "Detection deleted."]);
} else {
    echo json_encode(["success" => false, "message" => "Database error."]);
}
$delete->close();
$conn->close();
