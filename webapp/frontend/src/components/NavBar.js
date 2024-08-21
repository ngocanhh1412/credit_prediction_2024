// import React, { useState } from 'react';
// import './NavBar.css';

// const NavBar = ({ onSelect }) => {
//   const [activeButton, setActiveButton] = useState('');

//   const handleSelect = (selection) => {
//     setActiveButton(selection);
//     onSelect(selection);
//   };

//   return (
//     <div className="sidebar">
//       <div className="logo">
//         <span className="logo-luna">LUNA</span><span className="logo-dot">.</span><span className="logo-credit">Credit</span>
//       </div>

//       <hr className='hr-navbar'/>

//       <button 
//         className={activeButton === 'manual' ? 'isActive' : ''} 
//         onClick={() => handleSelect('manual')}
//       >
//         Nhập tay
//       </button>

//       <button 
//         className={activeButton === 'table' ? 'isActive' : ''} 
//         onClick={() => handleSelect('table')}
//       >
//         Bảng
//       </button>

//       <button 
//         className={activeButton === 'creditById' ? 'isActive' : ''} 
//         onClick={() => handleSelect('creditById')}
//       >
//         Credit Score by ID
//       </button>      
//     </div>
//   );
// };

// export default NavBar;

import React, { useState } from 'react';
import './NavBar.css';

const NavBar = ({ onSelect }) => {
  const [activeButton, setActiveButton] = useState('');

  const handleSelect = (selection) => {
    setActiveButton(selection);
    onSelect(selection);
  };

  return (
    <div className="sidebar">
      <button 
        className="logo-button" 
        onClick={() => handleSelect('homePage')}
      >
        <span className="logo-luna">LUNA</span><span className="logo-dot">.</span><span className="logo-credit">Credit</span>
      </button>

      <hr className='hr-navbar'/>

      <button 
        className={`sidebar-button ${activeButton === 'manual' ? 'isActive' : ''}`} 
        onClick={() => handleSelect('manual')}
      >
        Nhập tay
      </button>

      <button 
        className={`sidebar-button ${activeButton === 'table' ? 'isActive' : ''}`} 
        onClick={() => handleSelect('table')}
      >
        Bảng
      </button>

      <button 
        className={`sidebar-button ${activeButton === 'creditById' ? 'isActive' : ''}`} 
        onClick={() => handleSelect('creditById')}
      >
        Credit Score by ID
      </button>      
    </div>
  );
};

export default NavBar;
