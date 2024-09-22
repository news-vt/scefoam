// src/main.jsx
import React from 'react';
import { createRoot } from 'react-dom/client';
import Receiver from './Receiver';

const rootElement = document.getElementById('root');
createRoot(rootElement).render(<Receiver />);
