# CMake generated Testfile for 
# Source directory: /Users/yadnyesh8250/Desktop/VAJRA-OPT
# Build directory: /Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test("test_linalg" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/test_linalg")
set_tests_properties("test_linalg" PROPERTIES  WORKING_DIRECTORY "/Users/yadnyesh8250/Desktop/VAJRA-OPT" _BACKTRACE_TRIPLES "/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;111;add_test;/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;0;")
add_test("test_simplex" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/test_simplex")
set_tests_properties("test_simplex" PROPERTIES  WORKING_DIRECTORY "/Users/yadnyesh8250/Desktop/VAJRA-OPT" _BACKTRACE_TRIPLES "/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;116;add_test;/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;0;")
add_test("test_io" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/test_io")
set_tests_properties("test_io" PROPERTIES  WORKING_DIRECTORY "/Users/yadnyesh8250/Desktop/VAJRA-OPT" _BACKTRACE_TRIPLES "/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;121;add_test;/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;0;")
add_test("test_regression" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/test_regression")
set_tests_properties("test_regression" PROPERTIES  WORKING_DIRECTORY "/Users/yadnyesh8250/Desktop/VAJRA-OPT" _BACKTRACE_TRIPLES "/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;126;add_test;/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;0;")
add_test("test_presolve" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/test_presolve")
set_tests_properties("test_presolve" PROPERTIES  WORKING_DIRECTORY "/Users/yadnyesh8250/Desktop/VAJRA-OPT" _BACKTRACE_TRIPLES "/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;131;add_test;/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;0;")
add_test("test_pdhg" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/test_pdhg")
set_tests_properties("test_pdhg" PROPERTIES  WORKING_DIRECTORY "/Users/yadnyesh8250/Desktop/VAJRA-OPT" _BACKTRACE_TRIPLES "/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;136;add_test;/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;0;")
add_test("test_benchmark" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/indus_benchmark" "--out-dir" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/benchmark_artifacts" "--csv" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/benchmark_results.csv")
set_tests_properties("test_benchmark" PROPERTIES  WORKING_DIRECTORY "/Users/yadnyesh8250/Desktop/VAJRA-OPT" _BACKTRACE_TRIPLES "/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;153;add_test;/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;0;")
add_test("test_python_verifier" "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/validator/independent_verifier.py" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps" "/Users/yadnyesh8250/Desktop/VAJRA-OPT/build-cuda-requested/benchmark_artifacts/afiro.sol")
set_tests_properties("test_python_verifier" PROPERTIES  DEPENDS "test_benchmark" WORKING_DIRECTORY "/Users/yadnyesh8250/Desktop/VAJRA-OPT" _BACKTRACE_TRIPLES "/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;158;add_test;/Users/yadnyesh8250/Desktop/VAJRA-OPT/CMakeLists.txt;0;")
