<?php 
$host = "localhost";
$user = "root";
$password ="";
$db = "login";

mysql_connect($host, $user, $password);
mysql_select_db($db);

if (isset($_POST['username'])) {
    $uname = $_POST['username'];
    $password = $_POST['password'];

    $sql = "select * from loginform where user='".$uname."' AND Pass='".$password."' limit 1";

    $result =mysql_query($sql);

    if(mysql_num_rows($result)==1) {
        echo " You have successfully logged in";
        //exit();
    }
    else {
        echo "Incorrect username or password!";
       // exit();
    }

}


?>


<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css" integrity="sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk" crossorigin="anonymous">
  <link rel="stylesheet" href="css/style.css">
  <title>Log In</title>
</head>

<body>
  <div class="nav-container">
    <div class="wrapper">
      <nav>
        <div class="logo">
          TSF
        </div>


        <ul class="nav-items">

          <li>
            <a id=main href="main.html">main</a>
          </li>


          <li>
            <a id=main href="upload.html">upload</a>
          </li>



          <li>
            <a id=charts href="#">charts</a>
          </li>

          <li>
            <a class="nav-btn-container" href="#">
              <a id=models href="#">models</a>
              <img class="search-btn" src="images/search-icon.svg" alt="" />
              <img class="close-btn" src="images/close-icon.svg" alt="" />
            </a>
          </li>

        </ul>
      </nav>
    </div>
  </div>
  <div class="container my-5 d-flex justify-content-center">
    <div class="col-sm-5">
      <div class="card">
        <div class="card-header">Log In</div>
        <div class="card-body">
          <form id="form" method= "POST" action="#">
            <div class="form-group">
              <label for="username">Username</label>
              <input type="text" class="form-control" id="username" placeholder="Enter username">
              <div></div>
            </div>


            <div class="form-group">
              <label for="password">Password</label>
              <input type="password" class="form-control" id="password" placeholder="Enter password">
              <div></div>
            </div>

            <button type="submit" class="btn btn-primary btn-block">Log In</button>
            <li>
              <a id=signin href="signin.html">Don't have an account? : Sign in</a>
            </li>
            <li>
              <a id=without-signin href="dashboard.html">Continue without an account</a>
            </li>
          </form>
        </div>
      </div>
    </div>
  </div>


  <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
  <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js" integrity="sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI" crossorigin="anonymous"></script>
  <script src="login.js"></script>
</body>

</html>
