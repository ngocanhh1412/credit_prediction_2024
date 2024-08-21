import React, { useState } from 'react';
import Navbar from './components/NavBar';
import Form from './components/Form';
import Table from './components/Table';
import PredictByID from './components/PredictByID';
import HomePage from './components/HomePage';

import './App.css';

const App = () => {
  const [selectedSection, setSelectedSection] = useState('');

  const renderSection = () => {
    switch (selectedSection) {
      case 'manual':
        return <Form />; 
      case 'table':
        return <Table />; 
      case 'creditById':
        return <PredictByID />; 
      default:
        return <HomePage />;
    }
  };

  return (
    <div className="app">
      <Navbar onSelect={setSelectedSection} />
      <div className="content">
        {renderSection()}
      </div>
    </div>
  );
};

export default App;
