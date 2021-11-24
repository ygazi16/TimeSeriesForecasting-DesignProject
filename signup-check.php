<?php 
session_start(); 
$host = "localhost";
$user = "root";
$password ="dexter";
$db = "login";


$con= mysqli_connect($host, $user, $password);
mysqli_select_db($con, $db) or die($connect_error);

if (isset($_POST['uname']) && isset($_POST['password'])
    && isset($_POST['Email']) && isset($_POST['Phone']) && isset($_POST['re_password'])) {

	function validate($data){
       $data = trim($data);
	   $data = stripslashes($data);
	   $data = htmlspecialchars($data);
	   return $data;
	}

	$uname = validate($_POST['uname']);
	$pass = validate($_POST['password']);

	$re_pass = validate($_POST['re_password']);
	$phone = validate($_POST['Phone']);
    $email = validate($_POST['Email']);

	$user_data = 'uname='. $uname;


	if (empty($uname)) {
		header("Location: signin.php?error=User Name is required&$user_data");
	    exit();
	}else if(empty($pass)){
        header("Location: signin.php?error=Password is required&$user_data");
	    exit();
	}
	else if(empty($re_pass)){
        header("Location: signin.php?error=Re Password is required&$user_data");
	    exit();
	}

	else if(empty($phone)){
        header("Location: signin.php?error=Phone is required&$user_data");
	    exit();
	}

    else if(empty($email)){
        header("Location: signin.php?error=Email is required&$user_data");
	    exit();
	}

	else if($pass !== $re_pass){
        header("Location: signin.php?error=The confirmation password  does not match&$user_data");
	    exit();
	}

	else{

		// hashing the password
       // $pass = md5($pass);

	    $sql = "SELECT * FROM loginform WHERE User='$uname' ";
		$result = mysqli_query($con, $sql);

		if (mysqli_num_rows($result) > 0) {
			header("Location: signin.php?error=username is taken try another&$user_data");
	        exit();
		}else {
           $sql2 = "INSERT INTO loginform(User, Pass, Email, Phone) VALUES('$uname', '$pass', '$email', '$phone')";
           $result2 = mysqli_query($con, $sql2);
           if ($result2) {
           	 header("Location: signin.php?success=Your account has been created successfully");
	         exit();
           }else {
	           	header("Location: signin.php?error=unknown error occurred&$user_data");
		        exit();
           }
		}
	}
	
}else{
	header("Location: signin.php");
	exit();
}