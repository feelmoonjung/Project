<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%
    // 세션에서 user 객체 가져오기
    Object loginUser = session.getAttribute("loginUser");
%>


<!DOCTYPE html>
<html>

<head>
  <!-- Basic -->
  <meta charset="utf-8" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <!-- Mobile Metas -->
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
  <!-- Site Metas -->
  <meta name="keywords" content="" />
  <meta name="description" content="" />
  <meta name="author" content="" />

  <title>분석하기</title>

  <!-- bootstrap core css -->
  <link rel="stylesheet" type="text/css" href="css/bootstrap.css" />

  <!-- fonts style -->
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
  <!--owl slider stylesheet -->
  <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/OwlCarousel2/2.3.4/assets/owl.carousel.min.css" />
  <!-- nice select -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jquery-nice-select/1.1.0/css/nice-select.min.css" integrity="sha256-mLBIhmBvigTFWPSCtvdu6a76T+3Xyt+K571hupeFLg4=" crossorigin="anonymous" />
  <!-- font awesome style -->
  <link href="css/font-awesome.min.css" rel="stylesheet" />

  <!-- Custom styles for this template -->
  <link href="css/style.css?after" rel="stylesheet"/>
  <!-- responsive style -->
  <link href="css/responsive.css" rel="stylesheet" />
  
  <!-- java script -->
  <script type="text/javascript" src="/js/custom.js"></script>
  <!-- <script type="text/javascript">
	$(window).load(function() {
		$('#loading').hide();   
	});
  </script>  -->
  
  <style>
	.footer_section {
  		position: bottom; /* 화면 하단에 고정 */
  		bottom: 0;
  		left: 0;
  		width: 100%;
  		color: blue; /* 텍스트 색상 */
  		text-align: center;
  		padding: 5px 0; /* 상하 패딩을 작게 설정 */
 		box-shadow: none; /* 그림자 효과 제거 */
  		background-color: transparent; /* 배경색을 없앰 */
  		z-index: 1000; /* 다른 요소 위에 표시되도록 설정 */
		}
	#loading {
		 width: 100%;   
		 height: 100%;   
		 top: 0px;
		 left: 0px;
		 position: fixed;   
		 display: block;   
		 opacity: 0.7;   
		 background-color: #fff;   
		 z-index: 2000;   
		 text-align: center; }  
		 
	#loading-image {   
		 position: absolute;   
		 top: 50%;   
		 left: 50%;  
		 z-index: 2001; }	
		 
  </style>
</head>
<body class="sub_page">
  <!-- 로딩중 화면 -->
  <div id="loading" class = "loading"><img id="loading-image" src="./images/loading.gif" alt="Loading" /></div>
  
  <div class="hero_area">
    <!-- header section strats -->
    <header class="header_section">
      <div class="header_top">
        <div class="container-fluid header_top_container">
          
          <div class="contact_nav">
            <a href="A01.html">
              <i class="fa fa-map-marker" aria-hidden="true"></i>
              <span>
                Location
              </span>
            </a>
            <a href="BA02.html">
              <i class="fa fa-phone" aria-hidden="true"></i>
              <span>
                전화번호 : 031-224-3636
              </span>
            </a>
            <a href="BA02.html">
              <i class="fa fa-envelope" aria-hidden="true"></i>
              <span>
                Algo@gmail.com
              </span>
            </a>
            <!-- Display "로그아웃" if the user is logged in, otherwise "관리자" -->
            <% if (loginUser != null) { %>
              <a href="logout.jsp">
                <i class="fa fa-sign-out" aria-hidden="true"></i>
                <span>로그아웃</span>
              </a>
            <% } else { %>
              <a href="login.jsp">
                <i class="fa fa-user" aria-hidden="true"></i>
                <span>관리자</span>
              </a>
            <% } %>
          </div> 
        </div>
      </div>
      
      <div class="header_bottom">
        <div class="container-fluid">
          <nav class="navbar navbar-expand-lg custom_nav-container ">
            <div class="collapse navbar-collapse" id="navbarSupportedContent">
              <ul class="navbar-nav">
                <li class="nav-item active">
                  <a class="nav-link" href="main.jsp"><span><img src="images/logo.png" alt="logo" width = "77px"height="37px"></span></a>
                </li>
                <li class="nav-item">
                  <a class="nav-link" href="A01.jsp">회사소개</a>
                </li>
                
                <!-- 공지사항 링크-->
                <li class="nav-item">
                    <a class="nav-link" href="CA01.jsp">공지사항</a> <!-- 비로그인 시 CA01.jsp로 이동 -->
                </li>    
                
                
                <li class="nav-item">
                  <a class="nav-link" href="AB_main.jsp">분석사례</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="BA01.jsp">분석</a>
                </li>
                <li class="nav-item">
  					<% if (loginUser == null) { %>
    					<a class="nav-link" href="BA02.jsp">문의하기</a>
  					<% } %>
				</li>                   
                <!-- Show "마이페이지" next to "공지사항" if the user is logged in -->
                <% if (loginUser != null) { %>
                <li class="nav-item">
                  <a class="nav-link" href="DA01.jsp">마이페이지</a>
                </li>
                <% } %>
              </ul>
            </div>
          </nav>
        </div>
      </div>
    </header>
    <!-- end header section -->
  </div>


  <!-- ana section -->
  <section class="ana_section layout_padding">
    <div class="container">
      <div class="heading_container heading_center">
        <h2 style="color: white; font-weight: bold;">분석</h2>
      </div>
      <div class="row justify-content-center"> 
        <div class="col-md-13 px-0">
          <div class="form_container" style="background-color: rgb(255, 255, 255, 0.3); border-radius: 3px;">
            <form method = 'post' enctype="multipart/form-data" id = "analysis">             
              <div class = 'form-row'>
                    <input type="text" class="form-control" placeholder="email" required name = 'ana_email'/> <!-- 이메일 입력 -->
                    <select id="" class="form-control" required name = 'ana_type'> <!-- 분석타입 선택 -->
                      <option >분석 종류 (선택)</option>
                      <option value="공정최적화">공정최적화</option>
                      <option value="예지보전">예지보전</option>
                      <option value="품질보증">품질보증</option>
                    </select>
                </div>
                <div class = 'data-upload'>
                      <input class = 'upload-name' value="첨부파일(.csv)" disabled = 'disabled'>
                      <label for = 'file'>파일찾기</label>
                      <input type = 'file' id = 'file' class = 'upload-hidden' accept=".csv" name = 'ana_data' required/> <!-- 데이터 첨부 -->
                </div>
                  <div class="btn_box">
                    <button class = 'ana_btn' style = "margin-top : 10px; margin-right : 20px" onclick = "javascript : form.action='Analysis';">
                      분석하기
                    </button>
                    <button class = 'email_btn' style = "margin-top : 10px" onclick = "location.href = 'BA02.jsp'">
                      문의하기
                    </button>
                  
                  </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  </section>
  <!-- end ana section -->

 
  <!-- footer section -->
  <footer class="footer_section">
    <div class="container">
      <p>
        &copy; <span id="displayYear"></span> All Rights Reserved By
        <a href="main.jsp">AlgoRhythm</a>
      </p>
    </div>
  </footer>
  <!-- footer section -->

  <!-- jQery -->
  <script src="js/jquery-3.4.1.min.js"></script>
  <!-- popper js -->
  <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
  <!-- bootstrap js -->
  <script src="js/bootstrap.js"></script>
  <!-- owl slider -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/OwlCarousel2/2.3.4/owl.carousel.min.js"></script>
  <!--  OwlCarousel 2 - Filter -->
  <script src="https://huynhhuynh.github.io/owlcarousel2-filter/dist/owlcarousel2-filter.min.js"></script>
  <!-- nice select -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-nice-select/1.1.0/js/jquery.nice-select.min.js" integrity="sha256-Zr3vByTlMGQhvMfgkQ5BtWRSKBGa2QlspKYJnkjZTmo=" crossorigin="anonymous"></script>
  <!-- custom js -->
  <script src="js/custom.js"></script>
  <script type = text/javascript>
  $(document).ready(function(){
      var fileTarget = $('.data-upload .upload-hidden');

      fileTarget.on('change', function(){
          if(window.FileReader){
              var filename = $(this)[0].files[0].name;
          }
          else{
              var filename = $(this).val().split('/').pop().split('\\').pop();
          }

          $(this).siblings('.upload-name').val(filename);
      });
  });
  </script>
  <!-- fileupload js -->
  <script>
	  $(document).ready(function() {
	      $('#loading').hide();
	      $('#analysis').submit(function(){
	          $("#loading").css({
	              "top": (($(window).height()-$("#loading").outerHeight())/2+$(window).scrollTop())+"px",
	              "left": (($(window).width()-$("#loading").outerWidth())/2+$(window).scrollLeft())+"px"
	           }); 
	          $('#loading').show();
	          return true;
	      });
	  });
  </script>
</body>
</html>