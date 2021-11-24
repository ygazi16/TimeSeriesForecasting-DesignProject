<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css" integrity="sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk" crossorigin="anonymous">
  <link rel="stylesheet" href="css/style.css">
  <title>Sign In</title>
	
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
     <form action="signup-check.php" method="post">
     	<h2>SIGN UP</h2>
     	<?php if (isset($_GET['error'])) { ?>
     		<p class="error"><?php echo $_GET['error']; ?></p>
     	<?php } ?>

          <?php if (isset($_GET['success'])) { ?>
               <p class="success"><?php echo $_GET['success']; ?></p>
          <?php } ?>

          
          <label>User Name</label>
          <?php if (isset($_GET['uname'])) { ?>
               <input type="text" 
                      name="uname" 
                      placeholder="User Name"
                      value="<?php echo $_GET['uname']; ?>"><br>
          <?php }else{ ?>
               <input type="text" 
                      name="uname" 
                      placeholder="User Name"><br>
          <?php }?>


     	<label>Password</label>
     	<input type="password" 
                 name="password" 
                 placeholder="Password"><br>

        <label>Email</label>
          <?php if (isset($_GET['Email'])) { ?>
               <input type="text" 
                      name="Email" 
                      placeholder="Email"
                      value="<?php echo $_GET['Email']; ?>"><br>
          <?php }else{ ?>
               <input type="text" 
                      name="Email" 
                      placeholder="Email"><br>

          <?php }?>
          
          <label>Phone</label>
          <?php if (isset($_GET['Phone'])) { ?>
               <input type="text" 
                      name="Phone" 
                      placeholder="Phone"
                      value="<?php echo $_GET['Phone']; ?>"><br>
          <?php }else{ ?>
               <input type="text" 
                      name="Phone" 
                      placeholder="Phone"><br>
          <?php }?>

          <label>Re Password</label>
          <input type="password" 
                 name="re_password" 
                 placeholder="Re_Password"><br>

                 

     	<button type="submit">Sign Up</button>
          <a href="login.php" class="ca">Already have an account?</a>
     </form>
</body>
</html>