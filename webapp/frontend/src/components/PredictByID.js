import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Modal from 'react-modal';
import { useDropzone } from 'react-dropzone';
import './PredictByID.css';

Modal.setAppElement('#root'); // Set the root element for accessibility

const PredictByID = () => {
  const [memberId, setMemberId] = useState('');
  const [file, setFile] = useState(null);
  const [customers, setCustomers] = useState([]);
  const [notFoundIds, setNotFoundIds] = useState([]);
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [modalIsOpen, setIsOpen] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);
  const [error, setError] = useState('');
  const [threshold, setThreshold] = useState(null); // Thêm state cho threshold
  const [activeButton, setActiveButton] = useState('manual');

  // Lấy giá trị threshold từ API khi component được mount
  useEffect(() => {
    const fetchThreshold = async () => {
      try {
        const response = await axios.get('http://localhost:8000/threshold'); // Gọi API mới
        setThreshold(parseFloat(response.data.threshold)); // Lưu threshold vào state
      } catch (error) {
        console.error('Error fetching threshold:', error);
      }
    };
    fetchThreshold();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    setFile(acceptedFiles[0]);
  }, []);

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  const handleNavButtonClick = (buttonType) => {
    setActiveButton(buttonType);
    setIsUploaded(buttonType === 'csv');
    resetUpload(); // Reset when switching modes
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://localhost:8000/predict_by_id/predict_by_id', {
        member_id: parseInt(memberId)  // Chuyển đổi memberId thành số nguyên
      });
      const result = response.data;
      setSelectedCustomer(result);
      setError('');
    } catch (error) {
      if (error.response && error.response.status === 404) {
        setError('Không tìm thấy ID khách hàng');
      } else {
        console.error('Error fetching customer data:', error);
      }
    }
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/predict_by_id/bulk_predict_by_id', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      const { results, not_found_ids } = response.data;
      setCustomers(results);
      setNotFoundIds(not_found_ids);
      setIsUploaded(true);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleRowClick = (customer) => {
    setSelectedCustomer(customer);
    setIsOpen(true);
  };

  const closeModal = () => {
    setIsOpen(false);
  };

  const resetUpload = () => {
    setIsUploaded(false);
    setCustomers([]);
    setNotFoundIds([]);
    setFile(null);
    setSelectedCustomer(null);
    setMemberId('');
  };

  // Hàm xác định class dựa trên giá trị điểm và threshold
  const getScoreClassName = (score) => {
    return score < threshold ? 'id-span-score-red' : 'id-span-score-green';
  };

  return (
    <div className="predict-by-id-container">
      <div className="nav-buttons">
        <button 
          className={`nav-buttons-button ${activeButton === 'manual' ? 'active' : ''}`}
          onClick={() => handleNavButtonClick('manual')}
        >
          Nhập tay
        </button>
        <button 
          className={`nav-buttons-button ${activeButton === 'csv' ? 'active' : ''}`}
          onClick={() => handleNavButtonClick('csv')}
        >
          CSV
        </button>
      </div>

      {activeButton === 'manual' && (
        <div className="input-section">
          <h2>Nhập Member ID</h2>
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              className='input-id-text'
              value={memberId}
              onChange={(e) => setMemberId(e.target.value)}
              placeholder="Nhập Member ID"
            />
            <button className='form-submit-button' type="submit">Dự đoán</button>
          </form>

          {error && <p className="error-message">{error}</p>}

          {/* Hiển thị kết quả ngay dưới form */}
          {selectedCustomer && activeButton === 'manual' && error === '' && (
            <div className="result-section">
              <p className='p-table-modal'><strong>Tên:</strong> {selectedCustomer.name}</p>
            <p className='p-table-modal'><strong>ID khách hàng:</strong> {selectedCustomer.member_id}</p>
            <p className='p-table-modal'>
              <strong>Điểm tín dụng:</strong>{" "}
              <span className={getScoreClassName(selectedCustomer.score)}>
                {selectedCustomer.score}
              </span>
            </p>
            <h4 className='p-table-modal'>Thông tin chi tiết:</h4>
            <p><strong>Số kỳ thanh toán của khoản vay:</strong> {selectedCustomer.term}</p>
            <p><strong>Lãi suất của khoản vay:</strong> {selectedCustomer.int_rate}</p>
            <p><strong>Xếp hạng rủi ro tín dụng:</strong> {selectedCustomer.grade}</p>
            <p><strong>Số năm làm việc:</strong> {selectedCustomer.emp_length}</p>
            <p><strong>Tình trạng sở hữu nhà:</strong> {selectedCustomer.home_ownership}</p>
            <p><strong>Thu nhập hàng năm do người vay tự khai báo:</strong> {selectedCustomer.annual_inc}</p>
            <p><strong>Tình trạng xác minh thu nhập:</strong> {selectedCustomer.verification_status}</p>
            <p><strong>Lý do vay:</strong> {selectedCustomer.purpose}</p>
            <p><strong>Tỷ lệ nợ trên thu nhập:</strong> {selectedCustomer.dti}</p>
            <p><strong>Số lần người vay đã yêu cầu kiểm tra tín dụng trong 6 tháng gần nhất:</strong> {selectedCustomer.inq_last_6mths}</p>
            <p><strong>Tỷ lệ sử dụng hạn mức tín dụng:</strong> {selectedCustomer.revol_util}</p>
            <p><strong>Số tiền gốc còn lại chưa thanh toán:</strong> {selectedCustomer.out_prncp}</p>
            <p><strong>Tổng số tiền đã được thanh toán:</strong> {selectedCustomer.total_pymnt}</p>
            <p><strong>Tổng số tiền lãi đã được thu:</strong> {selectedCustomer.total_rec_int}</p>
            <p><strong>Tổng số tiền thanh toán cuối cùng:</strong> {selectedCustomer.last_pymnt_amnt}</p>
            <p><strong>Tổng số dư hiện tại của tất cả tài khoản:</strong> {selectedCustomer.tot_cur_bal}</p>
            <p><strong>Tổng hạn mức tín dụng quay vòng cao nhất:</strong> {selectedCustomer.total_rev_hi_lim}</p>
            <p><strong>Thời điểm dòng tín dụng đầu tiên được mở:</strong> {selectedCustomer.earliest_cr_line}</p>
            <p><strong>Tháng khoản vay được cấp vốn:</strong> {selectedCustomer.issue_d}</p>
            <p><strong>Tháng gần nhất nhận được khoản thanh toán từ người vay:</strong> {selectedCustomer.last_pymnt_d}</p>
            <p><strong>Tháng gần nhất kiểm tra báo cáo tín dụng cho khoản vay này:</strong> {selectedCustomer.last_credit_pull_d}</p>
            </div>
          )}
        </div>
      )}

      {activeButton === 'csv' && !isUploaded && (
        <div>
          <h2>Tải lên file CSV</h2>
          <div className='upload-section-container'>
            <div className="upload-section2">
              <div {...getRootProps({ className: 'dropzone' })}>
                <input {...getInputProps()} />
                <p>Kéo và thả file vào đây, hoặc nhấn để chọn file</p>
              </div>
              {file && (
                <div className="file-info">
                  <p>File đã chọn: {file.name}</p>
                  <button className='file-upload-button' onClick={handleUpload}>Dự đoán</button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {isUploaded && customers.length > 0 && (
        <>
          <div className="table-section">
            <button className="reset-button" onClick={resetUpload}>Dự đoán lại</button>
            <table>
              <thead>
                <tr>
                  <th>Tên</th>
                  <th>ID khách hàng</th>
                  <th>Điểm tín dụng</th>
                </tr>
              </thead>
              <tbody>
                {customers.map((customer) => (
                  <tr key={customer.member_id} onClick={() => handleRowClick(customer)}>
                    <td>{customer.name}</td>
                    <td>{customer.member_id}</td>
                    <td>
                      <span className={getScoreClassName(customer.score)}>
                        {customer.score}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {notFoundIds.length > 0 && (
            <div className="not-found-section">
              <h3>Các ID khách hàng không được tìm thấy:</h3>
              <ul>
                {notFoundIds.map((id) => (
                  <li key={id}>{id}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {/* Modal chỉ hiển thị cho chức năng CSV */}
      {activeButton === 'csv' && (
        <Modal
        isOpen={modalIsOpen}
        onRequestClose={closeModal}
        contentLabel="Customer Details Modal"
        className="modal"
        overlayClassName="overlay"
      >
        {selectedCustomer && (
          <div className="modal-content">
            <p className='p-table-modal'><strong>Tên:</strong> {selectedCustomer.name}</p>
            <p className='p-table-modal'><strong>ID khách hàng:</strong> {selectedCustomer.member_id}</p>
            <p className='p-table-modal'>
              <strong>Điểm tín dụng:</strong>{" "}
              <span className={getScoreClassName(selectedCustomer.score)}>
                {selectedCustomer.score}
              </span>
            </p>
            <h4 className='p-table-modal'>Thông tin chi tiết:</h4>
            <p><strong>Số kỳ thanh toán của khoản vay:</strong> {selectedCustomer.term}</p>
            <p><strong>Lãi suất của khoản vay:</strong> {selectedCustomer.int_rate}</p>
            <p><strong>Xếp hạng rủi ro tín dụng:</strong> {selectedCustomer.grade}</p>
            <p><strong>Số năm làm việc:</strong> {selectedCustomer.emp_length}</p>
            <p><strong>Tình trạng sở hữu nhà:</strong> {selectedCustomer.home_ownership}</p>
            <p><strong>Thu nhập hàng năm do người vay tự khai báo:</strong> {selectedCustomer.annual_inc}</p>
            <p><strong>Tình trạng xác minh thu nhập:</strong> {selectedCustomer.verification_status}</p>
            <p><strong>Lý do vay:</strong> {selectedCustomer.purpose}</p>
            <p><strong>Tỷ lệ nợ trên thu nhập:</strong> {selectedCustomer.dti}</p>
            <p><strong>Số lần người vay đã yêu cầu kiểm tra tín dụng trong 6 tháng gần nhất:</strong> {selectedCustomer.inq_last_6mths}</p>
            <p><strong>Tỷ lệ sử dụng hạn mức tín dụng:</strong> {selectedCustomer.revol_util}</p>
            <p><strong>Số tiền gốc còn lại chưa thanh toán:</strong> {selectedCustomer.out_prncp}</p>
            <p><strong>Tổng số tiền đã được thanh toán:</strong> {selectedCustomer.total_pymnt}</p>
            <p><strong>Tổng số tiền lãi đã được thu:</strong> {selectedCustomer.total_rec_int}</p>
            <p><strong>Tổng số tiền thanh toán cuối cùng:</strong> {selectedCustomer.last_pymnt_amnt}</p>
            <p><strong>Tổng số dư hiện tại của tất cả tài khoản:</strong> {selectedCustomer.tot_cur_bal}</p>
            <p><strong>Tổng hạn mức tín dụng quay vòng cao nhất:</strong> {selectedCustomer.total_rev_hi_lim}</p>
            <p><strong>Thời điểm dòng tín dụng đầu tiên được mở:</strong> {selectedCustomer.earliest_cr_line}</p>
            <p><strong>Tháng khoản vay được cấp vốn:</strong> {selectedCustomer.issue_d}</p>
            <p><strong>Tháng gần nhất nhận được khoản thanh toán từ người vay:</strong> {selectedCustomer.last_pymnt_d}</p>
            <p><strong>Tháng gần nhất kiểm tra báo cáo tín dụng cho khoản vay này:</strong> {selectedCustomer.last_credit_pull_d}</p>
            <button className='table-close-modal-button' onClick={closeModal}>Đóng</button>
          </div>
        )}
      </Modal>
      )}
    </div>
  );
};

export default PredictByID;
