<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css" integrity="sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk" crossorigin="anonymous">
    <link rel="stylesheet" href="style.css">
    <title>Sign In</title>
</head>
<body>
 <div id="frm">
    <form action="do.php" method="POST">
        <p >
            <label >username:</label>
            <input type="text" id="user" name="user">

        </p>

        <p >
            <label >password:</label>
            <input type="password" id="pass" name="pass">

        </p>
        <p>
            <input type="submit" id ="button" value="submit">
        </p>
       
    </form>

 </div>
</body>
</html>