import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './HomePage.css';

const HomePage = () => {
  const [threshold, setThreshold] = useState(null);

  useEffect(() => {
    const fetchThreshold = async () => {
      try {
        const response = await axios.get('http://localhost:8000/threshold'); // Đảm bảo backend có route này
        setThreshold(parseFloat(response.data.threshold));
      } catch (error) {
        console.error('Error fetching threshold:', error);
      }
    };

    fetchThreshold();
  }, []);

  return (
    <div className="homepage-container">
      <h1 className='homepage-container-h1 '>Đây là ứng dụng dự đoán điểm tín dụng khách hàng</h1>
      {threshold !== null ? (
        <p className='homepage-container-p'>Ngưỡng điểm đề xuất: <strong className='homepage-container-strong'>{threshold}</strong></p>
      ) : (
        <p className='homepage-container-p'>Đang tải ngưỡng điểm...</p>
      )}
    </div>
  );
};

export default HomePage;
