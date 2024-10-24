package common;

import java.io.IOException;


import javax.servlet.ServletException;
import javax.servlet.http.HttpSession;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.*;
import javax.servlet.http.*;

import java.io.File;
import java.sql.*;

import com.oreilly.servlet.MultipartRequest;
import com.oreilly.servlet.multipart.DefaultFileRenamePolicy;




/**
 * Servlet implementation class Analysis
 */
@WebServlet("/Analysis")
public class Analysis extends HttpServlet {
	private static final long serialVersionUID = 1L;
	AnalysisDao AnaDao = new AnalysisDao();
       
    /**
     * @see HttpServlet#HttpServlet()
     */
    public Analysis() {
        super();
//        AnalysisDao = new AnalysisDao(); //초기화 메소드 
        // TODO Auto-generated constructor stub
    }

	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		// TODO Auto-generated method stub
//		doGet(request, response);
		
//		request.setCharacterEncoding("UTF-8");
		
		// db에 email, type, path 업로드
		request.setCharacterEncoding("UTF-8");
		response.setContentType("text/html; charset=utf-8");
		
		String savePath = "C:\\savepath";
		int sizeLimit = 2024*2024*15;
		
		MultipartRequest mr = new MultipartRequest(request, savePath, sizeLimit, "UTF-8", new DefaultFileRenamePolicy());
		
		String ana_email = mr.getParameter("ana_email");
		String ana_type = mr.getParameter("ana_type");
		String ana_data = mr.getFilesystemName("ana_data");
		String ana_data_path = savePath + "/" + ana_data;
		
		AnaDo AnaDo = new AnaDo();
		AnaDo.setAna_email(ana_email);
		AnaDo.setAna_type(ana_type);
		AnaDo.setAna_data(ana_data); // 파일 이름
		AnaDo.setAna_path(ana_data_path); //경로 + 파일 이름
		
		// 데이터 입력 확인
		if (ana_email == null || ana_email.isEmpty()) {
            response.getWriter().println("<script>alert('이메일을 입력해 주세요.'); history.back();</script>");
            return;
        }
		if (ana_type == null || ana_type.isEmpty()) {
            response.getWriter().println("<script>alert('분석 종류를 선택해주세요.'); history.back();</script>");
            return;
        }
        if (ana_data == null || ana_data.isEmpty()) {
            response.getWriter().println("<script>alert('분석 데이터를 첨부해주세요.'); history.back();</script>");
            return;
        }
        // 유효성 검사 - 이메일 형식 확인
        if (!ana_email.matches("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+$")) {
            response.getWriter().println("<script>alert('유효한 이메일 형식을 입력해 주세요.'); history.back();</script>");
            return;
        }
		
		try{
			AnaDao.db_upload(AnaDo);
			
		}
		catch(SQLException e) {
			System.out.println("업로드 실패");
			e.printStackTrace();
			}
		// 분석 실행
		
		if(ana_type.equals("품질보증")) {
			// 데이터 일치 여부 확인
			if (!ana_data.contains("Ana_data_qc")) {
	            response.getWriter().println("<script>alert('데이터 형식과 종류가 일치하지 않습니다.'); history.back();</script>");
	            return;}
		File batFile = new File("C:\\python\\Ana_bat_qc.bat");
        ProcessBuilder processBuilder = new ProcessBuilder(batFile.getAbsolutePath());
        processBuilder.directory(new File("C:\\python\\"));

        try {
            Process process = processBuilder.start();

            new Thread(() -> {
                try (var reader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println(line);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }).start();

            int exitCode = process.waitFor();
            System.out.println("Process exited with code: " + exitCode);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
		}
		else if(ana_type.equals("예지보전")) {
			// 데이터 일치 여부 확인
			if (!ana_data.contains("Ana_data_yj")) {
	            response.getWriter().println("<script>alert('데이터 형식과 종류가 일치하지 않습니다.'); history.back();</script>");
	            return;}
			File batFile = new File("C:\\python\\Ana_bat_yj.bat");
	        ProcessBuilder processBuilder = new ProcessBuilder(batFile.getAbsolutePath());
	        processBuilder.directory(new File("C:\\python\\"));

	        try {
	            Process process = processBuilder.start();

	            new Thread(() -> {
	                try (var reader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))) {
	                    String line;
	                    while ((line = reader.readLine()) != null) {
	                        System.out.println(line);
	                    }
	                } catch (IOException e) {
	                    e.printStackTrace();
	                }
	            }).start();

	            int exitCode = process.waitFor();
	            System.out.println("Process exited with code: " + exitCode);
	        } catch (IOException | InterruptedException e) {
	            e.printStackTrace();
	        }
		}
		
		else if(ana_type.equals("공정최적화")) {
			// 데이터 일치 여부 확인
			if (!ana_data.contains("Ana_data_gj")) {
	            response.getWriter().println("<script>alert('데이터 형식과 종류가 일치하지 않습니다.'); history.back();</script>");
	            return;}
			File batFile = new File("C:\\python\\Ana_bat_gj.bat");
	        ProcessBuilder processBuilder = new ProcessBuilder(batFile.getAbsolutePath());
	        processBuilder.directory(new File("C:\\python\\"));

	        try {
	            Process process = processBuilder.start();

	            new Thread(() -> {
	                try (var reader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))) {
	                    String line;
	                    while ((line = reader.readLine()) != null) {
	                        System.out.println(line);
	                    }
	                } catch (IOException e) {
	                    e.printStackTrace();
	                }
	            }).start();

	            int exitCode = process.waitFor();
	            System.out.println("Process exited with code: " + exitCode);
	        } catch (IOException | InterruptedException e) {
	            e.printStackTrace();
	        }
		}
		else if(ana_type.equals("분석 종류 (선택)")) {
			response.getWriter().println("<script>alert('분석 종류를 선택해주시기 바랍니다.'); history.back();</script>");
            return;
		}
        
        // 결과 불러오기 및 출력
          try{
        	  String date = AnaDao.db_date(AnaDo);
        	  String accuracy = AnaDao.db_accuracy(AnaDo);
          	  String f1 = AnaDao.db_f1(AnaDo);
          	  String precision = AnaDao.db_precision(AnaDo);
          	  String recall = AnaDao.db_recall(AnaDo);
          	  String ana_model = AnaDao.db_model(AnaDo);
          	  String image_path = AnaDao.db_image(AnaDo);
          	  
          	  String email = AnaDo.getAna_email();
          	  
          	  // 결과 파라미터 출력
    		  System.out.println(date);
          	  System.out.println(accuracy);
    		  System.out.println(f1);
    		  System.out.println(precision);
    		  System.out.println(recall);
    		  System.out.println(ana_model);
    		  System.out.println(image_path);
//    		  HttpSession session = request.getSession();
    		  
    		  // 결과 파라미터 set
    		  request.setAttribute("date", date);
    		  request.setAttribute("accuracy", accuracy);
    		  request.setAttribute("f1", f1);
    		  request.setAttribute("precision", precision);
    		  request.setAttribute("recall", recall);
    		  request.setAttribute("ana_model", ana_model);
    		  request.setAttribute("image_path", image_path);
    		
    		  
    		  // email 파라미터 set
    		  request.setAttribute("email", email);
    		  
    		  ServletContext app = this.getServletContext();
    		  RequestDispatcher dispatcher = app.getRequestDispatcher("/BA03.jsp");
    		  try {
    			  dispatcher.forward(request, response);
    		  } catch (ServletException e) {
    			  System.out.println(e);
    		  }
    		  
		  }
		  catch(SQLException e) {
			  System.out.println("업로드 실패");
			  e.printStackTrace();
			  }
	}   
}
		
		

//      // parameter 확인		
//		System.out.println(AnaDo.getAna_email());
//		System.out.println(AnaDo.getAna_sel());
//		System.out.println(AnaDo.getAna_data());
//		System.out.println(AnaDo.getAna_path());
		
//		AnaDo ana_info = new AnaDo();
//		ana_info.setAna_email(ana_email);
//		ana_info.setAna_sel(ana_sel);
//		
//		File batFile = new File("C:\\Users\\mit106\\Desktop\\project\\JAVA\\WokrSpace\\PROJECT\\src\\main\\java\\PROJECT_test\\test.bat");
//        ProcessBuilder processBuilder = new ProcessBuilder(batFile.getAbsolutePath());
//        processBuilder.directory(new File("C:\\Users\\mit106\\Desktop\\project\\JAVA\\WokrSpace\\PROJECT\\src\\main\\java\\PROJECT_test\\"));
//
//        try {
//            Process process = processBuilder.start();
//
//            new Thread(() -> {
//                try (var reader = new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))) {
//                    String line;
//                    while ((line = reader.readLine()) != null) {
//                        System.out.println(line);
//                    	AnalysisDao.result_insert(line); // 추가 insert 코드
//                    }
//                } catch (IOException e) {
//                    e.printStackTrace();
//                } catch(SQLException e) {
//        			System.out.println("업로드 실패");
//        			e.printStackTrace();
//                }
//            }).start();
//            
//            int exitCode = process.waitFor();
//            System.out.println("Process exited with code: " + exitCode);
//
//        } catch (IOException | InterruptedException e) {
//            e.printStackTrace();
//        }		
