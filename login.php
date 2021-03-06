<!DOCTYPE html>
<html>
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
		<form action="do.php" method="post">
     	
     	<?php if (isset($_GET['error'])) { ?>
     		<p class="error"><?php echo $_GET['error']; ?></p>
     	<?php } ?>     	

     	<label for="username">Username</label>
              <input type="username" class="form-control" name= "uname" id="username" placeholder="Enter password">
              <div></div>

		 <label for="password">Password</label>
              <input type="password" class="form-control" name= "password" id="password" placeholder="Enter password">
              <div></div>

			  
			  <div> 
			  <button type="submit" class="btn btn-primary btn-block">Log In</button> </div>
            <li>
              <a id=signin href="signin.php">Don't have an account? : Sign in</a>
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