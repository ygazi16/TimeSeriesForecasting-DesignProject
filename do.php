<?php 
session_start();
$host = "localhost";
$user = "root";
$password ="dexter";
$db = "login";


$con= mysqli_connect($host, $user, $password);
mysqli_select_db($con, $db) or die($connect_error);

if (isset($_POST['uname'])) {
   
    
    $uname = $_POST['uname'];
    $password = $_POST['password'];

    $sql = "select * from loginform where User='".$uname."' AND Pass='".$password."' limit 1";

    $result =mysqli_query($con, $sql);

    if(mysqli_num_rows($result)==1) {
        echo "You have successfully logged in";
        exit();
    }
    else {
        echo "Incorrect username or password!";
        exit();
    }

    

}

else {
    echo "incorrect username or password!";
    exit();
}


?>