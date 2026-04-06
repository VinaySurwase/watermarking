import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import "./App.css";

import Login from "./pages/Login";
import Register from "./pages/Register";
import Upload from "./pages/Upload";
import Verify from "./pages/Verify";
import ExtractUpload from "./pages/Extractupload"
import Compare from "./pages/Compare"

function PrivateRoute({ children }) {
  const token = localStorage.getItem("token");

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

export default function App() {
  const token = localStorage.getItem("token");

  return (
    <BrowserRouter>
      <Routes>
        {/* Public */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected */}
        <Route
          path="/upload"
          element={
            <PrivateRoute>
              <Upload />
            </PrivateRoute>
          }
        />

        <Route
          path="/verify"
          element={
            <PrivateRoute>
              <Verify />
            </PrivateRoute>
          }
        />
        <Route
          path="/extract"
          element={
            <PrivateRoute>
              <ExtractUpload />
            </PrivateRoute>
          }
        />
        <Route
          path="/compare"
          element={
            <PrivateRoute>
              <Compare />
            </PrivateRoute>
          }
        />

        {/* Default Route */}
        <Route
          path="/"
          element={
            token ? <Navigate to="/upload" /> : <Navigate to="/login" />
          }
        />

        {/* Catch All */}
        <Route
          path="*"
          element={
            token ? <Navigate to="/upload" /> : <Navigate to="/login" />
          }
        />
      </Routes>
    </BrowserRouter>
  );
}