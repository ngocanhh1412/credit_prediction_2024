import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Modal from 'react-modal';
import { useDropzone } from 'react-dropzone';
import './Table.css';

Modal.setAppElement('#root'); // Set the root element for accessibility

const Table = () => {
  const [file, setFile] = useState(null);
  const [customers, setCustomers] = useState([]);
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [modalIsOpen, setIsOpen] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);
  const [threshold, setThreshold] = useState(null); // Thêm state cho threshold

  // Fetch the threshold value from threshold.txt
  useEffect(() => {
    const fetchThreshold = async () => {
      try {
        const response = await axios.get('http://localhost:8000/threshold'); // Route mới cho threshold
        setThreshold(parseFloat(response.data.threshold));
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

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/table/bulk_predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      const results = response.data;
      setCustomers(results);
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
    setFile(null);
    setSelectedCustomer(null);
  };

  const getScoreClassName = (score) => {
    return score < threshold ? 'table-span-score-red' : 'table-span-score-green';
  };

  return (
    <div className="table-container">
      {!isUploaded ? (
        <div className="upload-section">
          <div {...getRootProps({ className: 'dropzone' })}>
            <input {...getInputProps()} />
            <p>Kéo và thả file CSV tại đây, hoặc nhấn để chọn file</p>
          </div>
          {file && (
            <div className="file-info">
              <p>File đã chọn: {file.name}</p>
              <button className='file-upload-button' onClick={handleUpload}>Dự đoán</button>
            </div>
          )}
        </div>
      ) : (
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
                  <td className={getScoreClassName(customer.score)}>{customer.score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
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
    </div>
  );
};

export default Table;
