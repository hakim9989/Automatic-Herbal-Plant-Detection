    <?php
    // login.php
    ini_set('display_errors', 1); // Temporarily enable for debugging
    ini_set('display_startup_errors', 1); // Temporarily enable for debugging
    error_reporting(E_ALL); // Temporarily enable for debugging

    header("Access-Control-Allow-Origin: *");
    header("Access-Control-Allow-Headers: Content-Type, Authorization");
    header("Access-Control-Allow-Methods: GET, POST, OPTIONS");

    if ($_SERVER['REQUEST_METHOD'] == 'OPTIONS') {
        http_response_code(200);
        exit();
    }

    session_start(); // Start session once
    header('Content-Type: application/json');
    require_once 'db_connect.php'; // Include database connection once

    if ($_SERVER["REQUEST_METHOD"] == "POST") {
        $input = file_get_contents("php://input");
        $data = json_decode($input, true);

        $userName = $data['userName'] ?? '';
        $password = $data['password'] ?? '';

        if (empty($userName) || empty($password)) {
            echo json_encode(["success" => false, "message" => "Username and password are required."]);
            exit();
        }

        $sql = "SELECT id, userName, password_hash FROM users WHERE userName = ?";

        if ($stmt = $conn->prepare($sql)) {
            $stmt->bind_param("s", $userName);

            if ($stmt->execute()) {
                $stmt->store_result();

                if ($stmt->num_rows == 1) {
                    $stmt->bind_result($id, $userName_db, $password_hash_db);
                    $stmt->fetch();

                    if (password_verify($password, $password_hash_db)) {
                        // Authentication successful
                        // Generate a new unique token
                        $authToken = bin2hex(openssl_random_pseudo_bytes(32)); // 32 bytes for 64 hex chars token

                        // Store token in the database
                        $insertTokenSql = "INSERT INTO auth_tokens (user_id, auth_token, expires_at) VALUES (?, ?, ?)";
                        $expiryTime = date('Y-m-d H:i:s', strtotime('+1 hour')); // Token expires in 1 hour

                        if ($insertStmt = $conn->prepare($insertTokenSql)) {
                            $insertStmt->bind_param("iss", $id, $authToken, $expiryTime);
                            if ($insertStmt->execute()) {
                                // Token stored successfully, return it to the client
                                error_log("LOGIN SUCCESS: User ID: $id | UserName: $userName_db | Auth Token: $authToken (DB stored)");
                                echo json_encode([
                                    "success" => true,
                                    "message" => "Login successful!",
                                    "user_id" => $id,
                                    "user_name" => $userName_db,
                                    "auth_token" => $authToken // Return the token to Flutter
                                ]);
                            } else {
                                error_log("LOGIN ERROR: Failed to store auth token in DB: " . $insertStmt->error);
                                echo json_encode(["success" => false, "message" => "Login failed: Could not create session token."]);
                            }
                            $insertStmt->close();
                        } else {
                            error_log("LOGIN ERROR: Failed to prepare token insert statement: " . $conn->error);
                            echo json_encode(["success" => false, "message" => "Login failed: Database error."]);
                        }

                    } else {
                        error_log("LOGIN FAILED: Invalid password for user: $userName");
                        echo json_encode(["success" => false, "message" => "Invalid username or password."]);
                    }
                } else {
                    error_log("LOGIN FAILED: User not found: $userName");
                    echo json_encode(["success" => false, "message" => "Invalid username or password."]);
                }
            } else {
                error_log("LOGIN DB ERROR: " . $stmt->error);
                echo json_encode(["success" => false, "message" => "Error during login: " . $stmt->error]);
            }
            $stmt->close();
        } else {
            error_log("LOGIN PREPARE ERROR: " . $conn->error);
            echo json_encode(["success" => false, "message" => "Database query preparation failed: " . $conn->error]);
        }

        $conn->close();
    } else {
        echo json_encode(["success" => false, "message" => "Invalid request method."]);
    }
    // NO CLOSING PHP TAG HERE TO PREVENT ACCIDENTAL WHITESPACE
    