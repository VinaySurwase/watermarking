import { useNavigate, useLocation } from "react-router-dom";

export default function Topbar() {
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };

  return (
    <div className="topbar">
      <div className="topbar-left">
        <span className="logo" onClick={() => navigate("/upload")}>
          WatermarkX
        </span>
      </div>

      <div className="topbar-center">
        <button
          className={`nav-btn ${
            location.pathname === "/upload" ? "active" : ""
          }`}
          onClick={() => navigate("/upload")}
        >
          Upload
        </button>

        <button
          className={`nav-btn ${
            location.pathname === "/verify" ? "active" : ""
          }`}
          onClick={() => navigate("/verify")}
        >
          Verify
        </button>
         <button
          className={`nav-btn ${
            location.pathname === "/extract" ? "active" : ""
          }`}
          onClick={() => navigate("/extract")}
        >
          Extract
        </button>
        <button
          className={`nav-btn ${
            location.pathname === "/compare" ? "active" : ""
          }`}
          onClick={() => navigate("/compare")}
        >
          Compare
        </button>
      </div>

      <div className="topbar-right">
        <button className="logout-btn" onClick={handleLogout}>
          Logout
        </button>
      </div>
    </div>
  );
}
