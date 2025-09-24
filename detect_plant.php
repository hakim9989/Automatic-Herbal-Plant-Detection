<?php
// detect_plant.php
session_start();
header('Content-Type: application/json');
require_once 'db_connect.php';

// --- Debug logging ---
error_log("DETECT PLANT REQUEST: Session ID: " . session_id() . " | Method: " . $_SERVER['REQUEST_METHOD']);
error_log("DETECT PLANT REQUEST: Cookies: " . print_r($_COOKIE, true));
error_log("DETECT PLANT REQUEST: Session: " . print_r($_SESSION, true));

// Get the X-Auth-Token header
$headers = getallheaders();
$authToken = $headers['X-Auth-Token'] ?? null;
error_log("DETECT PLANT REQUEST: Received X-Auth-Token: " . ($authToken ?? 'NULL'));

// 1️⃣ Validate token and extend expiry if valid
$user_id = null;

if ($authToken) {
    $validateSql = "SELECT user_id, expires_at FROM auth_tokens WHERE auth_token = ?";
    if ($stmt = $conn->prepare($validateSql)) {
        $stmt->bind_param("s", $authToken);
        $stmt->execute();
        $stmt->store_result();

        if ($stmt->num_rows === 1) {
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
                    error_log("DETECT PLANT: Token expiry extended to $newExpiry for User ID: $user_id");
                } else {
                    error_log("DETECT PLANT: Failed to prepare expiry update: " . $conn->error);
                }

                // Optional: Update PHP session
                $_SESSION['loggedin'] = true;
                $_SESSION['id'] = $user_id;
                $_SESSION['auth_token'] = $authToken;
            } else {
                // Token expired → delete it
                $deleteSql = "DELETE FROM auth_tokens WHERE auth_token = ?";
                if ($delStmt = $conn->prepare($deleteSql)) {
                    $delStmt->bind_param("s", $authToken);
                    $delStmt->execute();
                    $delStmt->close();
                }
                error_log("DETECT PLANT: Token expired and removed: $authToken");
            }
        } else {
            error_log("DETECT PLANT: Token not found in DB");
        }

        $stmt->close();
    } else {
        error_log("DETECT PLANT: DB prepare error: " . $conn->error);
    }
}

// Fail if authentication failed
if ($user_id === null) {
    error_log("DETECT PLANT: Authentication failed (invalid or expired token).");
    http_response_code(401);
    echo json_encode(["success" => false, "message" => "Authentication required or session expired. Please login."]);
    exit();
}

// 2️⃣ Only allow POST
if ($_SERVER["REQUEST_METHOD"] !== "POST") {
    echo json_encode(["success" => false, "message" => "Invalid request method."]);
    exit();
}

// 3️⃣ Check uploaded image
if (!isset($_FILES['image']) || $_FILES['image']['error'] !== UPLOAD_ERR_OK) {
    echo json_encode(["success" => false, "message" => "No image uploaded or upload error."]);
    exit();
}

$image_tmp_name = $_FILES['image']['tmp_name'];
$image_file_name = $_FILES['image']['name'];

// 4️⃣ Forward to Flask API
$flask_api_url = "http://127.0.0.1:5000/predict";

$ch = curl_init();
curl_setopt($ch, CURLOPT_URL, $flask_api_url);
curl_setopt($ch, CURLOPT_POST, 1);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$cfile = new CURLFile($image_tmp_name, $_FILES['image']['type'], $image_file_name);
curl_setopt($ch, CURLOPT_POSTFIELDS, ['image' => $cfile]);

$flask_response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$curl_error = curl_error($ch);
curl_close($ch);

if ($curl_error) {
    error_log("DETECT PLANT CURL ERROR: $curl_error");
    http_response_code(500);
    echo json_encode(["success" => false, "message" => "Failed to connect to Flask API: $curl_error"]);
    exit();
}

if ($http_code !== 200) {
    error_log("DETECT PLANT FLASK ERROR: HTTP $http_code | Response: $flask_response");
    http_response_code($http_code);
    echo json_encode(["success" => false, "message" => "Flask API returned an error (HTTP $http_code): $flask_response"]);
    exit();
}

$prediction_data = json_decode($flask_response, true);
if (json_last_error() !== JSON_ERROR_NONE) {
    error_log("DETECT PLANT JSON ERROR: " . json_last_error_msg() . " | Response: $flask_response");
    http_response_code(500);
    echo json_encode(["success" => false, "message" => "Failed to parse Flask API response."]);
    exit();
}

if (!isset($prediction_data['plant_type'])) {
    error_log("DETECT PLANT: Missing 'plant_type' in response: $flask_response");
    http_response_code(500);
    echo json_encode(["success" => false, "message" => "Unexpected Flask API response format."]);
    exit();
}

// 5️⃣ Save detection to database
$plant_type = $prediction_data['plant_type'];
$confidence = $prediction_data['confidence'];
$scientific_name = $prediction_data['details']['scientific_name'] ?? null;
$description = $prediction_data['details']['description'] ?? null;
$medicinal_uses = $prediction_data['details']['medicinal_uses'] ?? null;

$insertSql = "INSERT INTO plant_detections (user_id, plant_type, confidence, scientific_name, description, medicinal_uses) VALUES (?, ?, ?, ?, ?, ?)";
if ($stmt = $conn->prepare($insertSql)) {
    $stmt->bind_param("isdsss", $user_id, $plant_type, $confidence, $scientific_name, $description, $medicinal_uses);
    if ($stmt->execute()) {
        error_log("DETECT PLANT: Saved detection for User ID: $user_id | Plant: $plant_type");
        echo json_encode(array_merge(["success" => true, "message" => "Plant detected and saved!"], $prediction_data));
    } else {
        error_log("DETECT PLANT DB ERROR: " . $stmt->error);
        echo json_encode(["success" => false, "message" => "Failed to save detection to database: " . $stmt->error]);
    }
    $stmt->close();
} else {
    error_log("DETECT PLANT PREPARE ERROR: " . $conn->error);
    echo json_encode(["success" => false, "message" => "Database error: " . $conn->error]);
}

$conn->close();
