<?php
    $username=$_POST['user'];
    $password=$_POST['pass'];

    $username=stripcslashes($username);
    $password=stripcslashes($password);

    

    $username = mysql_real_escape_string($username);
    $password = mysql_real_escape_string($password);

    mysql_connect("localhost","root","");
    mysql_select_db("signin");

    $result = mysql_query("select * from client where username = '$username' and password = '$password'"); 
         or die("failed, error".mysql_error());
    $row = mysql_fetch_array ($result);
    if($row['username']==$username && $row['password']==$password){
        echo "successfully signed in ! ".$row['username'];
    } else {
        echo "wrong password or username";
    }
?>



