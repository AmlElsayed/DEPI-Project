<!DOCTYPE html>
<html>
    <head>
        <title>predictive-care</title>
        <link rel="stylesheet" href="css/x-ray_style.css">
        <link rel="stylesheet" href="css/animate.css">
        <link rel="stylesheet" href="css/normalize.css">
        <link rel="stylesheet" href="css/all.min.css">

    </head>
    <body>
        <div class="chest">
            <div class="chest_form"><h1 class="chest_heading wow fadeInLeftBig" data-wow-delay="0.3s">PNEUMONIA-DETECTION</h1>

        <form class="wow fadeInLeftBig" data-wow-delay="0.3s"action="X-ray-img-upload.php" method="POST" enctype="multipart/form-data">
              <label>Upload sample image:</label>
            <input required type="file" name="img" accept=".jpg,.jpeg,.png">
            <input type="submit" name="submit-img" value="Upload image">
        </form> </div>

            </div>
            <script src="js/wow.min.js"></script>
        <script>
          new WOW().init();
          </script>
        
    </body>
</html>
 