import React, { useState } from 'react';
import Modal from 'react-modal';

import './Form.css';

const customStyles = {
  content: {
    top: '50%',
    left: '50%',
    right: 'auto',
    bottom: 'auto',
    marginRight: '-50%',
    transform: 'translate(-50%, -50%)',
  },
};

Modal.setAppElement('#root'); // Set the root element for accessibility

const Form = () => {
    const [formData, setFormData] = useState({
        term: '36',
        int_rate: '',
        grade: 'A',
        emp_length: '',
        home_ownership: 'OWN',
        annual_inc: '',
        verification_status: 'Source Verified',
        purpose: 'debt_consolidation',
        dti: '',
        inq_last_6mths: '',
        revol_util: '',
        out_prncp: '',
        total_pymnt: '',
        total_rec_int: '',
        last_pymnt_amnt: '',
        tot_cur_bal: '',
        total_rev_hi_lim: '',
        earliest_cr_line: '',
        issue_d: '',
        last_pymnt_d: '',
        last_credit_pull_d: ''
    });

    const [modalIsOpen, setIsOpen] = useState(false);
    const [creditScore, setCreditScore] = useState(null);
    const [threshold, setThreshold] = useState(null); // Thêm state cho ngưỡng điểm

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const payload = {
            ...formData,
            term: parseInt(formData.term),
            int_rate: parseFloat(formData.int_rate),
            annual_inc: parseFloat(formData.annual_inc),
            dti: parseFloat(formData.dti),
            inq_last_6mths: parseInt(formData.inq_last_6mths),
            revol_util: parseFloat(formData.revol_util),
            out_prncp: parseFloat(formData.out_prncp),
            total_pymnt: parseFloat(formData.total_pymnt),
            total_rec_int: parseFloat(formData.total_rec_int),
            last_pymnt_amnt: parseFloat(formData.last_pymnt_amnt),
            tot_cur_bal: parseFloat(formData.tot_cur_bal),
            total_rev_hi_lim: parseFloat(formData.total_rev_hi_lim)
        };

        const response = await fetch('http://localhost:8000/form/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        setCreditScore(result.score);
        setThreshold(result.threshold); // Lưu giá trị threshold từ backend
        setIsOpen(true);
    };

    const closeModal = () => {
        setIsOpen(false);
    };

    return (
        <div>
            <form className='input-form' onSubmit={handleSubmit}>
                <div className='div-form'>
                    <label>Số kỳ thanh toán của khoản vay</label>
                    <select className='input-select' name="term" value={formData.term} onChange={handleChange}>
                        <option value="36">36 tháng</option>
                        <option value="60">60 tháng</option>
                    </select>
                </div>
                <div className='div-form'>
                    <label>Xếp hạng rủi ro tín dụng</label>
                    <select className='input-select' name="grade" value={formData.grade} onChange={handleChange}>
                        <option value="A">A</option>
                        <option value="B">B</option>
                        <option value="C">C</option>
                        <option value="D">D</option>
                        <option value="E">E</option>
                        <option value="F">F</option>
                        <option value="G">G</option>
                    </select>
                </div>
                <div className='div-form'>
                    <label>Số năm làm việc</label>
                    <select className='input-select' name="emp_length" value={formData.emp_length} onChange={handleChange}>
                        <option value="< 1 year">Dưới 1 năm</option>
                        <option value="1 year">1 năm</option>
                        <option value="2 years">2 năm</option>
                        <option value="3 years">3 năm</option>
                        <option value="4 years">4 năm</option>
                        <option value="5 years">5 năm</option>
                        <option value="6 years">6 năm</option>
                        <option value="7 years">7 năm</option>
                        <option value="8 years">8 năm</option>
                        <option value="9 years">9 năm</option>
                        <option value="10+ years">10+ năm</option>
                    </select>
                </div>
                <div className='div-form'>
                    <label>Tình trạng sở hữu nhà</label>
                    <select className='input-select' name="home_ownership" value={formData.home_ownership} onChange={handleChange}>
                        <option value="OWN">Đã sở hữu</option>
                        <option value="MORTGAGE">Nhà trả góp</option>
                        <option value="RENT">Đang thuê nhà</option>
                        <option value="ANY">Khác</option>
                    </select>
                </div>
                <div className='div-form'>
                    <label>Tình trạng xác minh thu nhập</label>
                    <select className='input-select' name="verification_status" value={formData.verification_status} onChange={handleChange}>
                        <option value="Source Verified">Nguồn thu nhập đã được xác minh</option>
                        <option value="Verified">Thu nhập đã được xác minh bởi tổ chức cho vay</option>
                    </select>
                </div>
                <div className='div-form'>
                    <label>Lý do vay</label>
                    <select className='input-select' name="purpose" value={formData.purpose} onChange={handleChange}>
                        <option value="debt_consolidation">Hợp nhất các khoản nợ</option>
                        <option value="credit_card">Thanh toán nợ thẻ tín dụng</option>
                        <option value="major_purchase">Mua sắm </option>
                        <option value="car">Mua xe hơi</option>
                        <option value="home_improvement">Sửa sang nhà cửa</option>
                        <option value="educational">Chi phí giáo dục</option>
                        <option value="renewable_energy">Năng lượng tái tạo</option>
                        <option value="small_business">Vốn kinh doanh nhỏ</option>
                        <option value="moving">Chuyển nhà</option>
                        <option value="vacation">Chi phí kỳ nghỉ</option>
                        <option value="house">Mua nhà</option>
                        <option value="wedding">Đám cưới</option>
                        <option value="medical">Chi phí y tế</option>
                        <option value="other">Khác</option>
                    </select>
                </div>
                <div className='div-form'>
                    <label>Lãi suất của khoản vay</label>
                    <input className='input-text' type="text" name="int_rate" value={formData.int_rate} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Thu nhập hàng năm do người vay tự khai báo</label>
                    <input className='input-text' type="text" name="annual_inc" value={formData.annual_inc} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tỷ lệ nợ trên thu nhập</label>
                    <input className='input-text' type="text" name="dti" value={formData.dti} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Số lần người vay đã yêu cầu kiểm tra tín dụng trong 6 tháng gần nhất</label>
                    <input className='input-text' type="text" name="inq_last_6mths" value={formData.inq_last_6mths} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tỷ lệ sử dụng hạn mức tín dụng</label>
                    <input className='input-text' type="text" name="revol_util" value={formData.revol_util} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Số tiền gốc còn lại chưa thanh toán</label>
                    <input className='input-text' type="text" name="out_prncp" value={formData.out_prncp} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tổng số tiền đã được thanh toán</label>
                    <input className='input-text' type="text" name="total_pymnt" value={formData.total_pymnt} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tổng số tiền lãi đã được thu</label>
                    <input className='input-text' type="text" name="total_rec_int" value={formData.total_rec_int} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tổng số tiền thanh toán cuối cùng</label>
                    <input className='input-text' type="text" name="last_pymnt_amnt" value={formData.last_pymnt_amnt} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tổng số dư hiện tại của tất cả tài khoản</label>
                    <input className='input-text' type="text" name="tot_cur_bal" value={formData.tot_cur_bal} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tổng hạn mức tín dụng quay vòng cao nhất</label>
                    <input className='input-text' type="text" name="total_rev_hi_lim" value={formData.total_rev_hi_lim} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Thời điểm dòng tín dụng đầu tiên được mở</label>
                    <input className='input-text' type="text" name="earliest_cr_line" value={formData.earliest_cr_line} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tháng khoản vay được cấp vốn</label>
                    <input className='input-text' type="text" name="issue_d" value={formData.issue_d} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tháng gần nhất nhận được khoản thanh toán từ người vay</label>
                    <input className='input-text' type="text" name="last_pymnt_d" value={formData.last_pymnt_d} onChange={handleChange} />
                </div>
                <div className='div-form'>
                    <label>Tháng gần nhất kiểm tra báo cáo tín dụng cho khoản vay này</label>
                    <input className='input-text' type="text" name="last_credit_pull_d" value={formData.last_credit_pull_d} onChange={handleChange} />
                </div>
                <button className='form-submit-button' type="submit">Dự đoán</button>
            </form>
            <Modal
                className='form-modal'
                isOpen={modalIsOpen}
                onRequestClose={closeModal}
                contentLabel="Credit Score Modal"
            >
                <h2>
                    Điểm tín dụng: <span className={creditScore < threshold ? 'credit-score-span-red' : 'credit-score-span-green'}>{creditScore}</span>
                </h2>
                {/* <h3>Customer Information:</h3>
                <pre>{formatDataForDisplay(formData)}</pre> */}
                <button className='form-submit-button' onClick={closeModal}>Đóng</button>
            </Modal>
        </div>
    );
};
    
export default Form;